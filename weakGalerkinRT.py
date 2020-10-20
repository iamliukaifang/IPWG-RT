#!/usr/bin/python3
''' Purpose: WG - RT

Log:
====
    18/7/2020: Converge in L2 norm
        # todo: 
            error in h1-norm

'''

# import sys
# sys.path.append('/Users/kaifang/surfdrive/MyDocs/projects/pyMaxwell/lib')
# # print('Lib folder "%s" has been added to PATH.\n' % sys.path[-1])
import time

from basicmesh import BasicMesh
from femesh import PolygonMesh
import quads as qd
import tools as tl

import numpy as np
import numba as nb
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt


# import PDEs
import elliptic_equation_ex1 as pde



class WGMesh(PolygonMesh):
    '''
    Attributes:
    ===========
        degree - (1 by 1, int) degree of the RT element
        bdy_edges - (Nb by 1, int) global indices of boundary edges
    '''
    def __init__( self, basic_mesh, degree=1 ):
        PolygonMesh.__init__(self, basic_mesh)

        self.degree = degree
        # indices of boundary
        self.bdy_edges = np.argwhere(self.edge_faces[:,1]==-1)[:,0]
        self.__constrained_dof = None
        self.__free_dof = None

        # an array includes the local dof on all elements
        self.__dof = None

        self.A = None           # global stiffness matrix
        self.uh = None          # WG solution
        self.Qhu = None         # L2 projection of the exact solution u
        self.error_L2 = None    # relative error in L2-norm
        self.error_H1 = None    # relative error in H1-norm


    @property
    def dof(self):
        if self.__dof is None:
            k = self.degree
            self.__dof = self.face_verts.shape[0]*(k+1)*(k+2)//2 \
                + self.edge_verts.shape[0]*(k+1)
        return self.__dof


    def _update_constrained_dof(self):
        k = self.degree
        Nb = self.bdy_edges.shape[0]
        dof = np.zeros(Nb*(k+1), dtype=np.int64)
        dof0 = dim_Pk(k) * self.face_verts.shape[0]

        for i in range(Nb):
            ie = self.bdy_edges[i]
            ids = dof0 + ie*(k+1) + np.arange(k+1)
            ids_dof = i*(k+1) + np.arange(k+1)
            dof[ids_dof] = ids
        
        self.__constrained_dof = dof

    @property
    def constrained_dof(self):
        if self.__constrained_dof is None:
            self._update_constrained_dof()
        return self.__constrained_dof

    @property
    def free_dof(self):
        if self.__free_dof is None:
            self.__free_dof = np.setdiff1d(np.arange(self.dof),self.constrained_dof)
        return self.__free_dof


    def plot_solution(self):
        '''
        '''
        from mayavi import mlab

        Nt = self.face_verts.shape[0]
        x = np.zeros(3*Nt)
        y = np.zeros(3*Nt)
        z = np.zeros(3*Nt)

        k = self.degree
        dim0 = dim_Pk(k)
        
        for i in range( self.face_edges.shape[0] ):
            # coordinates of the element
            coords = self.verts[self.get_face_verts(i),:]
            # 
            ids = np.arange(3*i,3*(i+1))
            x[ids] = coords[:,0]
            y[ids] = coords[:,1]

            # global indices for uh
            u0_ids = np.arange(i*dim0, (i+1)*dim0)

            for j in range(3):
                phy0,_,_ = monomials(k, coords[j,:])
                z[ids[j]] = np.dot(self.uh[u0_ids], phy0)
                # z[ids[j]] = np.dot((self.Qhu[u0_ids]-self.uh[u0_ids])*50, phy0)

        mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        # Visualize the points
        pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=0.03)
        mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
        mlab.show()


#-- end of the class --#


@nb.njit
def dim_Pk(k):
    return (k+1)*(k+2)//2

@nb.njit
def dim_RT(k):
    return (k+1)*(k+3)


@nb.njit(nb.types.UniTuple(nb.float64[:],3)(nb.int64, nb.float64[:]))
def monomials(k, coords):
    ''' 
    Purpose: Compute the values and derivative values of monomials of degree K
        on the reference element ([0,0],[1,0],[0,1])
    Parameters:
        k	-	(i) Degree of monomial space
        coords	-	(i) 1 by 2. Coordinates of (quadrature) points
        V	-	(o, 1 by .5*(k+1)*(k+2)) values
        Vx	-	(o, 1 by .5*(k+1)*(k+2)) dx values
        Vy	-	(o, 1 by .5*(k+1)*(k+2)) dy values
    todo: change it to include Nedelec elements
    '''
    dim = np.int64(.5*(k+1)*(k+2))

    V = np.zeros(dim, dtype=np.float64)
    Vs = np.zeros(dim, dtype=np.float64)
    Vt = np.zeros(dim, dtype=np.float64)

    # monomial basis
    dssn, dttn = np.zeros(k+1), np.zeros(k+1)
    ssn = coords[0]**np.arange(k+1)
    ttn = coords[1]**np.arange(k+1)

    dssn[1:] = np.arange(1, k+1) * coords[0]**np.arange(k)
    dttn[1:] = np.arange(1, k+1) * coords[1]**np.arange(k)
    ii = 0
    for i in range(k+1):  # degree i 
        for j in range(i+1): # 
            V[ii] = ssn[i-j] * ttn[j]
            Vs[ii] = dssn[i-j] * ttn[j]
            Vt[ii] = ssn[i-j] * dttn[j]
            ii += 1
	
    return V, Vs, Vt


@nb.njit(nb.types.UniTuple(nb.float64[:],3)(nb.int64, nb.float64[:]))
def monomial_RT(k, coords):
    ''' Purpose: Compute the values and div values of 
        Raviart-Tomos (in monomials) of degree k: 
            P_k^2 + P_k * [X Y]':
            dim = (k+1)(k+2)+(k+1)
    Parameters:
        k	-	(i) Degree of homogeneous monomial space
        coords	-	(i, 1 by 2) Coordinates of (quadrature) points
        V	-	(o, .5*(k+1)*(k+2) by ) values
        Div	-	(o, 1 by .5*(k+1)*(k+2)) dx values
    '''
    dimPk = (k+1)*(k+2)//2
    dim = 2*dimPk + (k+1)

    V1 = np.zeros(dim, dtype=np.float64)
    V2 = np.zeros(dim, dtype=np.float64)
    Div = np.zeros(dim, dtype=np.float64)

    Pk, Pkx, Pky = monomials(k, coords)
    V1[:dimPk] = Pk
    V2[dimPk:2*dimPk] = Pk

    # subtract homo-monomial polynomials
    homoPk, homoPkx, homoPky = Pk[-k-1:], Pkx[-k-1:], Pky[-k-1:]
    V1[2*dimPk:] = homoPk * coords[0]
    V2[2*dimPk:] = homoPk * coords[1]

    Div[:dimPk] = Pkx
    Div[dimPk:2*dimPk] = Pky
    Div[2*dimPk:] = 2*homoPk + homoPkx*coords[0] + homoPky*coords[1]

    return V1, V2, Div


# @nb.njit(nb.float64[:](nb.int64, nb.float64[:,:]))
def get_exact_solution_elementwise(k, coords):
    ''' '''
    dim0 = dim_Pk(k)

    vol = tl.area_triangle(coords)
    qwts, qpts_bary_coords = qd.quad_triangle(1+k+k, vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)

    QM = np.zeros((dim0, dim0))
    Qb = np.zeros(dim0)
    for j in range(qwts.shape[0]):
        phy0, _, _ = monomials(k, qpts[j,:])
        QM += qwts[j]*np.outer(phy0, phy0)
        Qb += qwts[j]*pde.u(qpts[j,:1],qpts[j,1:2])*phy0
    
    uh = np.linalg.solve(QM, Qb)
    return uh


@nb.njit(nb.float64[:](nb.int64, nb.float64[:,:]))
def get_exact_solution_edgewise(k, edge_coords):
    ''' '''
    dim = k+1
    qwts_1d, qpts_1d = qd.quad_1d(1+2*k)

    le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])

    QM_2 = np.zeros((dim,dim))
    Qb_2 = np.zeros(dim)
    for j in range(qwts_1d.shape[0]):
        qxy = .5*( (edge_coords[1,:]-edge_coords[0,:])*qpts_1d[j]+\
            (edge_coords[1,:]+edge_coords[0,:]) )
        val_vb = qpts_1d[j]**(np.arange(k+1))
        QM_2 += .5*le* qwts_1d[j] * np.outer(val_vb, val_vb)
        Qb_2 += .5*le* qwts_1d[j] * pde.u(qxy[:1], qxy[1:2]) * val_vb
    
    uh = np.linalg.solve(QM_2, Qb_2)

    return uh



def get_exact_solution(T):
    '''
    '''
    dof = T.dof
    k = T.degree
    T.Qhu = np.zeros(dof)
    Nt = T.face_edges.shape[0]
    Ne = T.edge_verts.shape[0]
    dim0 = dim_Pk(k)

    for i in range(Nt):
        coords = T.verts[T.get_face_verts(i),:]
        u_ids = np.arange(i*dim0, (i+1)*dim0)
        T.Qhu[u_ids] = get_exact_solution_elementwise(k, coords)

    for i in range(Ne):
        edge_coords = T.verts[T.edge_verts[i,:],:]
        u_ids = Nt*dim0 + i*(k+1) + np.arange(k+1)
        T.Qhu[u_ids] = get_exact_solution_edgewise(k, edge_coords)


    return 0


@nb.njit
def DZT_RT(k, coords, os=np.array([True, True, True])):
    '''
    os
    '''
    center = np.sum(coords,0)/3
    vol = tl.area_triangle(coords)
    qwts, qpts_bary_coords = qd.quad_triangle(2*(k+1), vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)

    dimRT = (k+1)*(k+3)
    Dk = np.zeros((dimRT,dimRT))
    Zk = np.zeros((dimRT, (k+1)*(k+2)//2))
    Tk = np.zeros((dimRT, 3*(k+1)))

    for j in range(qwts.shape[0]):
        chi_1, chi_2, chi_div = monomial_RT(k, qpts[j,:] - center)
        phy0, _, _ = monomials(k, qpts[j,:])

        Dk += qwts[j]*(np.outer(chi_1, chi_1) + \
            np.outer(chi_2, chi_2))
        Zk += qwts[j] * np.outer(chi_div, phy0)

    # Tk = < φ_b_j, χ_i*n >_{\partial K}
    qwts_1d, qpts_1d = qd.quad_1d(k+1+k)

    # - loop over edges
    edges = np.array([[0,1],[1,2],[2,0]])
    for ie in range(3):
        edge_coords = coords[edges[ie,:],:]
        le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])
        n_1 = (edge_coords[1,1]-edge_coords[0,1])/le
        n_2 = (edge_coords[0,0]-edge_coords[1,0])/le

        cols = np.arange(ie*(k+1),(ie+1)*(k+1))
        for j in range(qwts_1d.shape[0]):
            qxy = .5*( (edge_coords[1,:]-edge_coords[0,:])*qpts_1d[j]+\
                (edge_coords[1,:]+edge_coords[0,:]) )
            chi_1, chi_2, _ = monomial_RT(k, qxy - center)
            if os[ie]:
                ub = qpts_1d[j]**np.arange(k+1)
            else:
                ub = (-qpts_1d[j])**np.arange(k+1)
            Tk[:,cols] += .5*le*qwts_1d[j] * np.outer(chi_1*n_1+chi_2*n_2, \
                ub)

    return Dk, Zk, Tk


def get_nnz(T):
    Nt = T.face_edges.shape[0]
    dim = dim_Pk(T.degree)+3*(T.degree+1)

    return Nt * dim**2


@nb.njit(nb.int64(nb.int64, nb.float64[:,:], nb.float64[:,:], nb.boolean[:]))
def locmat_vol( k, coords, Aloc, os ):
    '''
    Purpose: compute local matrices arising from volume 
    Parameters:
        coords - (i, float64, 3 by 2) coordinates of the triangle
        u_ord - (i, int64, 1 by 1) 
        Aloc - (o, float64, udim by udim) curl inner-product term
        L2loc - (o, float64, udim by udim) L2 inner-product term
        Floc - (o, float64, udim by 1) local rhs arising from volume
    Global variables:
    
    '''
    dimP0 = dim_Pk(k)
    
    Dk, Zk, Tk = DZT_RT(k, coords, os)
    Aloc[:dimP0, :dimP0] = tl.mat_mat( Zk.T, np.linalg.solve(Dk,Zk) )
    Aloc[:dimP0, dimP0:] = - tl.mat_mat( Zk.T, np.linalg.solve(Dk,Tk) )
    Aloc[dimP0:, :dimP0] = Aloc[:dimP0, dimP0:].T
    Aloc[dimP0:, dimP0:] = tl.mat_mat( Tk.T, np.linalg.solve(Dk,Tk) )

    return 0


@nb.njit(nb.int64(nb.int64, nb.float64[:,:], nb.float64[:]))
def locrhs_vol( k, coords, Floc ):
    '''
    Purpose: compute local matrices arising from volume 
    Parameters:
        coords - (i, float64, 3 by 2) coordinates of the triangle
        k - (i, int64, 1 by 1) 
        Floc - (o, float64, udim by 1) local rhs arising from volume
    '''
    vol = tl.area_triangle(coords)
    qwts, qpts_bary_coords = qd.quad_triangle(1+2*k, vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)
    
    for j in range(qwts.shape[0]):
        phy0, _, _ = monomials(k, qpts[j,:])
        Floc += qwts[j] * pde.f(qpts[j,:1],qpts[j,1:2])*phy0

    return 0



def global_system(T, rows_global, cols_global, data_global, Fglobal):
    '''
    Purpose: assemble the global matrix, which is in COO format.
    Parameters:
        data_global - (o, float64, nnz by 1)
        rows_global - (o, int64, nnz by 1)
        cols_global - (o, int64, nnz by 1) Global matrix
        Fglobal - (o) N by 1. Global right-hand-side
    '''
	
	# get the number of elements and edges
    Nt, Ne = T.face_edges.shape[0], T.edge_verts.shape[0]
	# Start position of non-zeros elements of global stiffness matrix in COO.
	# In general, Next need to be updated after putting each local matrix 
	# into the global one.
    Next = 0

    print('Computing local matrices and rhs arising from volume ...')
    # t_start = time.time()
    k = T.degree
    dim_u0 = dim_Pk(k)
    for i in range( Nt ):
		# coordinates of the element and its area
        coords = T.verts[T.get_face_verts(i),:]
        edges = T.face_edges[i,:]
        os = T.face_edge_orientations[i,:]
		
		# global indices
        u0_ids = np.arange(i*dim_u0, (i+1)*dim_u0)

        ub_ids = np.zeros(3*(k+1), dtype=np.int64)
        ub_ids[:k+1] = dim_u0*Nt + edges[0]*(k+1) + np.arange(k+1)
        ub_ids[k+1:2*(k+1)] = dim_u0*Nt + edges[1]*(k+1) + np.arange(k+1)
        ub_ids[2*(k+1):] = dim_u0*Nt + edges[2]*(k+1) + np.arange(k+1)
        # * Easy to make a mistake, the order of indices on each edge
        # * doesn't need to reverse
        # if not os[0]:
        #     ub_ids[:k+1] = np.flip(ub_ids[:k+1])
        # if not os[1]:
        #     ub_ids[k+1:2*(k+1)] = np.flip(ub_ids[k+1:2*(k+1)])
        # if not os[2]:
        #     ub_ids[2*(k+1):] = np.flip(ub_ids[2*(k+1):])

        u_ids = np.hstack((u0_ids, ub_ids))
		
        Aloc = np.zeros( ( u_ids.size, u_ids.size ) )
        # L2loc = np.zeros( ( u_ids.size, u_ids.size ) )
        Floc = np.zeros( dim_u0 )
        # compute i-th local stiff matrix and right-hand-side
        locmat_vol(k, coords, Aloc, os)
        locrhs_vol(k, coords, Floc)

        # put them in global matrix (in COO format)
        tl.array_to_coo( Aloc, u_ids, u_ids, \
            data_global, rows_global, cols_global, Next)
        Next += Aloc.size

        # assembling right-hand-side
        Fglobal[u0_ids] += Floc

    # t_end = time.time()
    # print('Done. Time elapsed: ', t_end-t_start )

    return 0


def weak_Galerkin_RT_solver( T ):

    Nt = T.face_edges.shape[0]

    # Store the global stiffness matrix in COO format
    # the number of non-zero element in coo format (upper bound)
    nnz_global = get_nnz(T)
    rows_global = np.zeros(nnz_global, dtype=np.int64)
    cols_global = np.zeros(nnz_global, dtype=np.int64)
    data_global = np.zeros(nnz_global)
    # T.show(labelFlag=False)
    # T.show_with_turtle(speed = 5)
    
    Fglobal = np.zeros(T.dof)	   # initialize global right-hand-side (rhs)

    global_system(T, rows_global, cols_global, data_global, Fglobal)

    # Transform triples into a sparse global stiffness matrix
    T.A = sp.csr_matrix((data_global, (rows_global, cols_global)), shape=(T.dof, T.dof))

    # Impose boundary condition
    T.uh = np.zeros(T.dof)
    T.uh[T.constrained_dof] = T.Qhu[T.constrained_dof]
    Fglobal -= T.A @ T.uh
    

    # plt.spy(T.A, marker=".")
    # plt.show()
    # print(np.linalg.cond(T.A.toarray()))

    # # # save matrix and rhs
    # sio.savemat('Aglobal_b6.mat', {'Aglobal':Aglobal, 'Fglobal':Fglobal})
    # exit(0)

    # # solve
    print('Solving global system ...')
    # t_start = time.time()
    T.uh[T.free_dof] = spla.spsolve((T.A[:,T.free_dof])[T.free_dof,:], Fglobal[T.free_dof])
    # t_end = time.time()
    # print('Elapsed time: ', t_end-t_start)

    return 0


def error_elementwise(k, coords, e0, eh, os):
    center = np.sum(coords,0)/3
    vol = tl.area_triangle(coords)
    qwts, qpts_bary_coords = qd.quad_triangle(1+2*k, vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)

    Dk, Zk, Tk = DZT_RT(k, coords, os)
    coe_weak_grads = np.linalg.solve(Dk, np.hstack((-Zk, Tk)))
    eh_weak_grads = np.dot(coe_weak_grads, eh)

    errors = np.zeros(2)

    for j in range(qwts.shape[0]):
        phy0, _, _ = monomials(k, qpts[j,:])
        errors[1] += qwts[j] * (np.dot(phy0, e0))**2

        chi_1, chi_2, _ = monomial_RT(k, qpts[j,:] - center )
        errors[0] += qwts[j] * (np.dot(chi_1, eh_weak_grads)**2 \
            + np.dot(chi_2, eh_weak_grads)**2)

    return errors


def weak_Galerkin_RT_errors(T):
    dof = T.dof
    Nt = T.face_edges.shape[0]
    Ne = T.edge_verts.shape[0]
    k = T.degree
    dim_u0 = dim_Pk(k)

    errors = np.zeros(2)
    norms = np.zeros(2)

    for i in range(Nt):
        coords = T.verts[T.get_face_verts(i),:]
        edges = T.face_edges[i,:]
        os = T.face_edge_orientations[i,:]
		
		# global indices
        u0_ids = np.arange(i*dim_u0, (i+1)*dim_u0)

        ub_ids = np.zeros(3*(k+1), dtype=np.int64)
        ub_ids[:k+1] = dim_u0*Nt + edges[0]*(k+1) + np.arange(k+1)
        ub_ids[k+1:2*(k+1)] = dim_u0*Nt + edges[1]*(k+1) + np.arange(k+1)
        ub_ids[2*(k+1):] = dim_u0*Nt + edges[2]*(k+1) + np.arange(k+1)

        u_ids = np.hstack((u0_ids, ub_ids))

        Qu0 = T.Qhu[u0_ids]
        Qu = T.Qhu[u_ids]
        e0 = T.Qhu[u0_ids]-T.uh[u0_ids]
        eh = T.Qhu[u_ids]-T.uh[u_ids]

        errors += error_elementwise(k, coords, e0, eh, os)
        norms += error_elementwise(k, coords, Qu0, Qu, os)

        T.error_H1, T.error_L2 = np.sqrt(errors/norms) # relative errors

    return 0



def main():
    for i in range(1,2):
        # Create triangle mesh
        # - mesh data .obj
        # fileName = "../mesh/triangles/triangle-2×4×4.obj"
        fileName = "../mesh/triangles/triangle-2x"+ str(2**i)+ "x" + str(2**i) + ".obj"

        bm = BasicMesh(fileName)
        T = WGMesh( bm, 1 )
        # T.show(face_label_flag=False)
        
        get_exact_solution(T)

        weak_Galerkin_RT_solver(T)

        weak_Galerkin_RT_errors(T)
        
        # # print("Errors: %11d %11.4e %11.4e \n" % (2**i, T.error_H1, T.error_L2))
        print("Errors: %11d %11.4e %11.4e" % (2**i, T.error_H1, T.error_L2))

        T.plot_solution() 


    print('------ end -------')



if __name__ == "__main__":
    main()
    # test()
    
