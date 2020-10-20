#!/usr/bin/python3
''' Purpose: IPWG - RT

Log:
====
    13/7/2020: converge with uniform degree 'k', but not for non-uniform 'k'
        # todo: 
            get error data

'''

import time
import numpy as np
import numba as nb
import scipy as sp
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt

# my libs
from lib.basicmesh import BasicMesh
from lib.femesh import PolygonMesh
import lib.quads as qd
import lib.tools as tl
from lib.tools import dim_Pk

# PDEs
import elliptic_equation_ex1 as pde


class WGMesh(PolygonMesh):
    '''
    Attributes:
    ===========
        local_dims - (Ne by 1, int) local degrees of freedom on each element
        start_pos - (Ne+1 by 1, int) global indices of the first 
            local basis functions on each element;
            For example, stat_pos[i]:start_pos[i+1] represents the 
            global indices of all shape functions on K_i
    '''
    def __init__( self, basic_mesh, degree=1 ):
        PolygonMesh.__init__(self, basic_mesh)

        self.degrees = degree * np.ones(self.face_edges.shape[0], dtype=np.int64)

        # an array includes the local dof on all elements
        self.__local_dims = None
        self.__start_pos = None

        self.A = None           # global stiffness matrix
        self.uh = None          # WG solution
        self.Qhu = None         # L2 projection of the exact solution u
        self.error_L2 = None
        self.error_H1 = None

    # Compute __local_dims
    def _update_local_dims(self):
        k = self.degrees
        self.__local_dims = (k+1)*(k+2)//2 + 3*(k+1)


    @property
    def local_dims(self):
        if self.__local_dims is None:
            self._update_local_dims()
        return self.__local_dims


    # Compute __local_dims
    def _update_start_pos(self):
        self.__start_pos = np.hstack((np.array([0],dtype=np.int64), np.cumsum(self.local_dims)))


    @property
    def start_pos(self):
        if self.__start_pos is None:
            self._update_start_pos()
        return self.__start_pos


    def plot_solution(self):
        '''
        '''
        from mayavi import mlab

        Nt = self.face_verts.shape[0]
        x = np.zeros(3*Nt)
        y = np.zeros(3*Nt)
        z = np.zeros(3*Nt)

        degrees = self.degrees
        local_dims = self.local_dims
        start_pos = self.start_pos
        
        for i in range( self.face_edges.shape[0] ):
            # coordinates of the element
            coords = self.verts[self.get_face_verts(i),:]
            center = np.sum(coords,0)/3.0
            # 
            ids = np.arange(3*i,3*(i+1))
            x[ids] = coords[:,0]
            y[ids] = coords[:,1]

            # global indices for uh
            dim0 = dim_Pk(degrees[i])
            u0_ids = np.arange(start_pos[i], start_pos[i]+dim0)

            for j in range(3):
                phy0,_,_ = monomials(degrees[i], coords[j,:]-center)
                z[ids[j]] = np.dot(self.Qhu[u0_ids], phy0)
                # z[ids[j]] = np.dot(self.uh[u0_ids], phy0)
                # z[ids[j]] = np.dot((self.Qhu[u0_ids]-self.uh[u0_ids])*50, phy0)

        mlab.figure(1, fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
        # Visualize the points
        pts = mlab.points3d(x, y, z, z, scale_mode='none', scale_factor=0.02)
        mlab.view(47, 57, 8.2, (0.1, 0.15, 0.14))
        mlab.show()


#-- end of the class --#





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
    dim = dim0+3*(k+1)

    uh = np.zeros(dim)

    center = np.sum(coords,0)/3
    vol = tl.area_triangle(coords)
    qwts, qpts_bary_coords = qd.quad_triangle(2*(k+1), vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)

    QM = np.zeros((dim0, dim0))
    Qb = np.zeros(dim0)

    for j in range(qwts.shape[0]):
        phy0, _, _ = monomials(k, qpts[j,:]-center)
        QM += qwts[j]*np.outer(phy0, phy0)
        Qb += qwts[j]*pde.u(qpts[j,:1],qpts[j,1:2])*phy0

    

    # - loop over edges
    qwts_1d, qpts_1d = qd.quad_1d(1+2*k)

    edges = np.array([[0,1],[1,2],[2,0]])
    for ie in range(3):
        edge_coords = coords[edges[ie,:],:]
        le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])

        cols = dim0 + ie*(k+1) + np.arange(k+1)
        
        QM_2 = np.zeros((k+1,k+1))
        Qb_2 = np.zeros(k+1)
        for j in range(qwts_1d.shape[0]):
            qxy = .5*( (edge_coords[1,:]-edge_coords[0,:])*qpts_1d[j]+\
                (edge_coords[1,:]+edge_coords[0,:]) )
            val_vb = qpts_1d[j]**(np.arange(k+1))
            QM_2 += .5*le* qwts_1d[j] * np.outer(val_vb, val_vb)
            Qb_2 += .5*le* qwts_1d[j] * pde.u(qxy[:1], qxy[1:2]) * val_vb
        
        uh[cols] = np.linalg.solve(QM_2, Qb_2)

    # print(np.linalg.cond(QM))
    uh[:dim0] = np.linalg.solve(QM, Qb)
    # lu, piv = sp.linalg.lu_factor(QM)
    # uh[:dim0] = sp.linalg.lu_solve((lu, piv), Qb)
    return uh


def get_exact_solution(T):
    '''
    '''
    dof = np.sum(T.local_dims)
    T.Qhu = np.zeros(dof)
    Nt = T.face_edges.shape[0]
    Ne = T.edge_verts.shape[0]
    for i in range(Nt):
        # print(i)
        coords = T.verts[T.get_face_verts(i),:]
        u_ids = np.arange(T.start_pos[i], T.start_pos[i+1])
        T.Qhu[u_ids] = get_exact_solution_elementwise(T.degrees[i], coords)

    return 0


@nb.njit
def DZT_RT(k, coords):
    '''
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
        phy0, _, _ = monomials(k, qpts[j,:]-center)

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
            Tk[:,cols] += .5*le*qwts_1d[j] * np.outer(chi_1*n_1+chi_2*n_2, \
                qpts_1d[j]**np.arange(k+1))

    return Dk, Zk, Tk


def get_nnz(T):
    Nt = T.face_edges.shape[0]
    Nv = T.verts.shape[0]
    Ne = T.edge_verts.shape[0]

    degrees = T.degrees
    dims = T.local_dims

    nnz = np.sum(dims**2)

    for i in range(Ne):
        if T.edge_faces[i,1] == -1:
            L = T.edge_faces[i,0]
            nnz += 2*dims[L]*(degrees[L]+1) + (degrees[L]+1)**2
        else:
            L, R = T.edge_faces[i,0], T.edge_faces[i,1]
            nnz += 2*(degrees[L]+degrees[R]+2)*(dims[L]+dims[R]) \
                + (degrees[L]+degrees[R]+2)**2

    return nnz


@nb.njit(nb.int64(nb.int64, nb.float64[:,:], nb.float64[:,:]))
def locmat_vol( k, coords, Aloc ):
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
    
    Dk, Zk, Tk = DZT_RT(k, coords)
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
    qwts, qpts_bary_coords = qd.quad_triangle(2*(k+1), vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)
    
    center = np.sum(coords,0)/3.
    for j in range(qwts.shape[0]):
        phy0, _, _ = monomials(k, qpts[j,:]-center)
        Floc += qwts[j] * pde.f(qpts[j,:1],qpts[j,1:2])*phy0

    return 0


# @nb.njit
@nb.njit(nb.int64(nb.float64[:,:], nb.int64, nb.int64, \
    nb.float64[:,:], nb.float64[:,:], nb.float64[:,:], nb.float64[:,:]))
def locmat_face( edge_coords, k_L, k_R, coords_L, coords_R, Muv, Mj ):
    '''
    Purpose: 
    ========
        compute local matrices arising from volume 
    Parameters:
    ===========
        coords - (i, float64, 3 by 2) coordinates of the triangle
        u_ord - (i, int64, 1 by 1) 
        Aloc - (o, float64, udim by udim) curl inner-product term
        L2loc - (o, float64, udim by udim) L2 inner-product term
        Floc - (o, float64, udim by 1) local rhs arising from volume
    '''
    le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])
    n_1 = (edge_coords[1,1]-edge_coords[0,1])/le
    n_2 = (edge_coords[0,0]-edge_coords[1,0])/le

    center_L, center_R = np.sum(coords_L,0)/3, np.sum(coords_R,0)/3

    Dk_L, Zk_L, Tk_L = DZT_RT(k_L, coords_L)
    Dk_R, Zk_R, Tk_R = DZT_RT(k_R, coords_R)
    # - coefficients of weak grad of basis 
    coe_L = np.linalg.solve(Dk_L, np.hstack((-Zk_L, Tk_L)))
    coe_R = np.linalg.solve(Dk_R, np.hstack((-Zk_R, Tk_R)))

    # - quadrature rule 1d
    qwts_1d, qpts_1d = qd.quad_1d(k_L+k_R+2)

    for j in range(qwts_1d.shape[0]):
        qxy = .5*( (edge_coords[1,:]-edge_coords[0,:])*qpts_1d[j]+\
            (edge_coords[1,:]+edge_coords[0,:]) )

        # - values of RT basis
        chi_L_1, chi_L_2, _ = monomial_RT(k_L, qxy - center_L)
        chi_R_1, chi_R_2, _ = monomial_RT(k_R, qxy - center_R)
        # - average of weak gradient, {grad_w uh * n_e}, of the two elements
        weak_grads = .5 * np.hstack((np.dot(chi_L_1*n_1+chi_L_2*n_2, coe_L),\
            np.dot(chi_R_1*n_1+chi_R_2*n_2, coe_R)))
        # - values of jumps [vb] on the edge
        val_vb = np.hstack((qpts_1d[j]**np.arange(k_L+1), -(-qpts_1d[j])**np.arange(k_R+1)))

        # 
        Muv += .5*le*qwts_1d[j] * np.outer(val_vb, weak_grads)
        Mj += .5*le*qwts_1d[j] * np.outer(val_vb, val_vb)
    
    return 0


@nb.njit(nb.int64(nb.float64[:,:], nb.int64, nb.float64[:,:],\
    nb.float64[:,:], nb.float64[:,:], nb.float64[:], nb.float64[:]))
def locmat_bdyface( edge_coords, k_L, coords_L, Muv, Mj, Floc1, Floc2 ):
    '''
    Purpose: 
    ========
        compute local matrices arising from volume 
    Parameters:
    ===========
        coords - (i, float64, 3 by 2) coordinates of the triangle
        k_L - (i, int64, 1 by 1) 
        Muv - (o, float64, udim by udim) curl inner-product term
    '''
    le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])
    n_1 = (edge_coords[1,1]-edge_coords[0,1])/le
    n_2 = (edge_coords[0,0]-edge_coords[1,0])/le

    center_L = np.sum(coords_L,0)/3

    Dk_L, Zk_L, Tk_L = DZT_RT(k_L, coords_L)
    # - coefficients of weak grad of basis 
    coe_L = np.linalg.solve(Dk_L, np.hstack((-Zk_L, Tk_L)))

    qwts_1d, qpts_1d = qd.quad_1d(1+k_L+k_L)

    # - to compute Qb(u) on the edge
    Qb = np.zeros(k_L+1)

    for j in range(qwts_1d.shape[0]):
        qxy = .5*( (edge_coords[1,:]-edge_coords[0,:])*qpts_1d[j]+\
            (edge_coords[1,:]+edge_coords[0,:]) )
        
        # - values of RT basis
        chi_L_1, chi_L_2, _ = monomial_RT(k_L, qxy - center_L)
        # - average of weak gradient, {grad_w uh * n_e}, of the one element
        weak_grads = np.dot(chi_L_1*n_1+chi_L_2*n_2, coe_L)
        # - values of jumps [vb] on the edge
        val_vb = qpts_1d[j]**np.arange(k_L+1)
        # 
        Muv += .5*le*qwts_1d[j] * np.outer(val_vb, weak_grads)
        Mj += .5*le*qwts_1d[j] * np.outer(val_vb, val_vb)

        # 
        Qb += .5*le*qwts_1d[j] * pde.u(qxy[:1],qxy[1:2]) * val_vb

    # - coefficient of Qb(u), note that Mj is the mass matrix on the edge
    Qbu = np.linalg.solve(Mj, Qb)

    Floc1 += tl.vec_mat(Qbu, Muv)
    Floc2 += tl.vec_mat(Qbu, Mj)

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
    start_pos = T.start_pos
    degrees = T.degrees
	
	# get the number of elements and edges
    Nt, Ne = T.face_edges.shape[0], T.edge_verts.shape[0]
	# Start position of non-zeros elements of global stiffness matrix in COO.
	# In general, Next need to be updated after putting each local matrix 
	# into the global one.
    Next = 0

    # print('Computing local matrices and rhs arising from volume ...')
    # t_start = time.time()
    for i in range( Nt ):
        k = degrees[i]
		# coordinates of the element and its area
        coords = T.verts[T.get_face_verts(i),:]
		
		# global indices
        dim_u0, dim_ub = dim_Pk(k), 3*(k+1)
        u0_ids = np.arange(start_pos[i], start_pos[i]+dim_u0, dtype=np.int64)
        u_ids = np.arange(start_pos[i], start_pos[i+1], dtype=np.int64)
		
        Aloc = np.zeros( ( u_ids.size, u_ids.size ) )
        # L2loc = np.zeros( ( u_ids.size, u_ids.size ) )
        Floc = np.zeros( dim_u0 )
        # compute i-th local stiff matrix and right-hand-side
        locmat_vol(k, coords, Aloc)
        locrhs_vol(k, coords, Floc)

        # put them in global matrix (in COO format)
        tl.array_to_coo( Aloc, u_ids, u_ids, \
            data_global, rows_global, cols_global, Next)
        Next += Aloc.size

        # assembling right-hand-side
        Fglobal[u0_ids] += Floc

    # t_end = time.time()
    # print('Done. Time elapsed: ', t_end-t_start )
	
    # print('Computing local matrices and rhs arising from faces ...')
    # t_start = time.time()
    for i in range(T.edge_verts.shape[0]):
        edge_coords = T.verts[T.edge_verts[i,:],:]
        le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])

        L = T.edge_faces[i,0]
        k_L = degrees[L]
        coords_L = T.verts[T.get_face_verts(L),:]
        u0_dim_L = dim_Pk(k_L)

        # - global indices of u on the left element
        u_ids_L = np.arange(start_pos[L], start_pos[L+1], dtype=np.int64)
        # - global indices of u on the edge (left element)
        edge_local_ids_L = T.edge_local_indices[i,0]
        ue_ids_L = start_pos[L]+u0_dim_L + \
            edge_local_ids_L * (k_L+1) + np.arange( k_L+1 )
        
        # boundary edges ...
        if T.edge_faces[i,1] == -1:
            Muv = np.zeros((ue_ids_L.size, u_ids_L.size))
            Mj = np.zeros((ue_ids_L.size, ue_ids_L.size))
            Floc1 = np.zeros(u_ids_L.size)
            Floc2 = np.zeros(ue_ids_L.size)

            locmat_bdyface( edge_coords, k_L, coords_L, Muv, Mj, Floc1, Floc2 )

            tl.array_to_coo( - Muv, ue_ids_L, u_ids_L, \
                data_global, rows_global, cols_global, Next)
            Next += Muv.size
            tl.array_to_coo( pde.EPSILON * Muv.T, u_ids_L, ue_ids_L, \
                data_global, rows_global, cols_global, Next)
            Next += Muv.size

            tl.array_to_coo( pde.SIGMA/(le**pde.BETA) * Mj, ue_ids_L, ue_ids_L, \
                data_global, rows_global, cols_global, Next)
            Next += Mj.size

            # rhs from boundary edges
            Fglobal[u_ids_L] += pde.EPSILON*Floc1
            Fglobal[ue_ids_L] += pde.SIGMA/(le**pde.BETA)*Floc2

        # interior edge
        else:
            R = T.edge_faces[i,1] 
            coords_R = T.verts[T.get_face_verts(R),:]
            k_R = degrees[R]
            
            u0_dim_R = dim_Pk(k_R)

            u_ids_R = np.arange(start_pos[R], start_pos[R+1], dtype=np.int64)
            # print(i, elems, 'u_ids_L', u_ids_L, u_ids_R)
            u_ids_LR = np.hstack((u_ids_L, u_ids_R))

            # - global indices of u on the edge (right element)
            edge_local_ids_R = T.edge_local_indices[i,1]
            ue_ids_R = start_pos[R] + u0_dim_R + \
                edge_local_ids_R * (k_R+1) + np.arange( k_R+1 )
            # - global indices of u on the edge (two elements)
            ue_ids_LR = np.hstack((ue_ids_L, ue_ids_R))
            
            Muv = np.zeros((ue_ids_LR.size, u_ids_LR.size))
            Mj = np.zeros((ue_ids_LR.size, ue_ids_LR.size))

            locmat_face( edge_coords, k_L, k_R, coords_L, coords_R, Muv, Mj )

            tl.array_to_coo( - Muv, ue_ids_LR, u_ids_LR, \
                data_global, rows_global, cols_global, Next)
            Next += Muv.size
            tl.array_to_coo( pde.EPSILON * Muv.T, u_ids_LR, ue_ids_LR, \
                data_global, rows_global, cols_global, Next)
            Next += Muv.size

            tl.array_to_coo( pde.SIGMA/(le**pde.BETA) * Mj, ue_ids_LR, ue_ids_LR, \
                data_global, rows_global, cols_global, Next)
            Next += Mj.size
            

    # t_end = time.time()
    # print('Time elapsed: ', t_end-t_start )
    return 0


def weak_Galerkin_RT_solver( T ):

    Nt = T.face_edges.shape[0]

    local_dims = T.local_dims
    # Store the global stiffness matrix in COO format
    # the number of non-zero element in coo format (upper bound)
    nnz_global = get_nnz(T)
    rows_global = np.zeros(nnz_global, dtype=np.int64)
    cols_global = np.zeros(nnz_global, dtype=np.int64)
    data_global = np.zeros(nnz_global)
    # T.show(labelFlag=False)
    # T.show_with_turtle(speed = 5)
    
    dof = np.sum(local_dims)   # global dof
    Fglobal = np.zeros(dof)	   # initialize global right-hand-side (rhs)

    global_system(T, rows_global, cols_global, data_global, Fglobal)

    # Transform triples into a sparse global stiffness matrix
    T.A = csc_matrix((data_global, (rows_global, cols_global)), shape=(dof, dof))

    # plt.spy(T.A, marker=".")
    # plt.show()
    # print(np.linalg.cond(T.A.toarray()))

    # # # save matrix and rhs
    # sio.savemat('Aglobal_b6.mat', {'Aglobal':Aglobal, 'Fglobal':Fglobal})
    # exit(0)

    # # solve
    # print('Solving global system ...')
    # t_start = time.time()
    T.uh = spsolve(T.A, Fglobal)
    # t_end = time.time()
    # print('Elapsed time: ', t_end-t_start)

    return 0


# @nb.njit(nb.float64[:](nb.int64, nb.float64[:,:],\
#     nb.float64[:], nb.float64[:]))
def error_H1_vol(k, coords, e0, eh):
    center  = np.sum(coords,0)/3.
    
    Dk, Zk, Tk = DZT_RT(k, coords )
    coe_weak_grads = np.linalg.solve(Dk, np.hstack((-Zk, Tk)))
    eh_weak_grads = np.dot(coe_weak_grads, eh)

    vol = tl.area_triangle(coords)
    qwts, qpts_bary_coords = qd.quad_triangle(2*(k+1), vol)
    qpts = tl.mat_mat(qpts_bary_coords, coords)

    errors = np.zeros(2) # h1 and l2 errors

    for j in range(qwts.shape[0]):
        phy0, _, _ = monomials(k, qpts[j,:] - center)
        errors[1] += qwts[j] * (np.dot(phy0, e0))**2

        chi_1, chi_2, _ = monomial_RT(k, qpts[j,:] - center )
        errors[0] += qwts[j] * (np.dot(chi_1, eh_weak_grads)**2 \
            + np.dot(chi_2, eh_weak_grads)**2)

    return errors


@nb.njit(nb.float64(nb.float64[:,:], nb.int64, nb.int64, \
    nb.float64[:], nb.float64[:]))
def error_H1_face(edge_coords, k_L, k_R, eb_L, eb_R):
    le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])

    eb = np.hstack((eb_L, eb_R))
    # - quadrature rule 1d
    qwts_1d, qpts_1d = qd.quad_1d(k_L+k_R+2)
    Mj = np.zeros((k_L+k_R+2, k_L+k_R+2))

    for j in range(qwts_1d.shape[0]):
        # - values of jumps [vb] on the edge
        val_vb = np.hstack((qpts_1d[j]**np.arange(k_L+1), -(-qpts_1d[j])**np.arange(k_R+1)))

        Mj += .5*le**(1-pde.BETA)*qwts_1d[j] * np.outer(val_vb, val_vb)
    
    return np.dot(eb, np.dot(Mj, eb))


# @nb.njit(nb.float64(nb.float64[:,:], nb.int64, nb.float64[:]))
def error_H1_bdyface(edge_coords, k_L, eb_L):
    le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])

    qwts_1d, qpts_1d = qd.quad_1d(1+k_L+k_L)

    Mj = np.zeros((k_L+1,k_L+1))

    for j in range(qwts_1d.shape[0]):
        # - values of jumps [vb] on the edge
        val_vb = qpts_1d[j]**np.arange(k_L+1)
        # 
        Mj += .5*le**(1-pde.BETA)*qwts_1d[j] * np.outer(val_vb, val_vb)

    return np.dot(eb_L, np.dot(Mj, eb_L))


def weak_Galerkin_RT_errors(T):
    '''Get the relative error in H1 and L2 norms  '''
    dof = np.sum(T.local_dims)
    Nt = T.face_edges.shape[0]
    Ne = T.edge_verts.shape[0]

    errors=np.zeros(2) # H1 and L2 errors
    norms = np.zeros(2) # H1 and L2 norms of Q_hu

    for i in range(Nt):
        coords = T.verts[T.get_face_verts(i),:]
        k = T.degrees[i]
        dim0 = dim_Pk(k)
        u0_ids = np.arange(T.start_pos[i], T.start_pos[i]+dim0)
        ub_ids = np.arange(T.start_pos[i]+dim0, T.start_pos[i+1])
        u_ids = np.arange(T.start_pos[i], T.start_pos[i+1])

        eh = T.Qhu[u_ids] - T.uh[u_ids]
        e0 = T.Qhu[u0_ids] - T.uh[u0_ids]

        # print(i, eh)

        errors += error_H1_vol(k, coords, e0, eh)
        norms += error_H1_vol(k, coords, T.Qhu[u0_ids], T.Qhu[u_ids])

        # print(i, errors, norms)

    for i in range(Ne):
        edge_coords = T.verts[T.edge_verts[i,:],:]
        le = np.linalg.norm(edge_coords[0,:]-edge_coords[1,:])
        L = T.edge_faces[i,0]
        k_L = T.degrees[L]

        u0_dim_L = dim_Pk(k_L)
        edge_local_ids_L = T.edge_local_indices[i,0]
        ue_ids_L = T.start_pos[L]+u0_dim_L + \
            edge_local_ids_L * (k_L+1) + np.arange( k_L+1 )

        eb_L = T.Qhu[ue_ids_L]-T.uh[ue_ids_L]
        ub_L = T.Qhu[ue_ids_L]
        
        if T.edge_faces[i,1] == -1:
            errors[0] += error_H1_bdyface(edge_coords, k_L, eb_L)
            norms[0] += error_H1_bdyface(edge_coords, k_L, ub_L)
        else:
            R = T.edge_faces[i,1] 
            k_R = T.degrees[R]
            
            u0_dim_R = dim_Pk(k_R)
            edge_local_ids_R = T.edge_local_indices[i,1]
            ue_ids_R = T.start_pos[R] + u0_dim_R + \
                edge_local_ids_R * (k_R+1) + np.arange( k_R+1 )
            
            eb_R = T.Qhu[ue_ids_R]-T.uh[ue_ids_R]
            ub_R = T.Qhu[ue_ids_R]

            # print(i, T.uh[ue_ids_L], T.uh[ue_ids_R], T.uh[ue_ids_L]-T.uh[ue_ids_R])
            # print(i, T.Qhu[ue_ids_L], T.Qhu[ue_ids_R], T.Qhu[ue_ids_L]-T.Qhu[ue_ids_R])

            errors[0] += error_H1_face(edge_coords, k_L, k_R, eb_L, eb_R)
            # norms[0] += error_H1_face(edge_coords, k_L, k_R, ub_L, ub_R)

        # print(i, errors, norms)

    return np.sqrt(errors)
    # return np.sqrt(errors)/np.sqrt(norms)


def main():
    for i in range(2,7):
        # Create triangle mesh
        # - mesh data .obj
        # fileName = "../mesh/triangles/triangle-2×4×4.obj"
        fileName = "./mesh/triangles/triangle-2x"+ str(2**i)+ "x" + str(2**i) + ".obj"

        # Load basic mesh consisting of vertices and faces
        basic_mesh = BasicMesh( fileName )

        T = WGMesh( basic_mesh, pde.order )
        # T.show(faceLabelFlag=True)
        
        get_exact_solution(T)

        weak_Galerkin_RT_solver(T)
        # T.plot_solution()

        # print(sp.sparse.linalg.norm(T.A - T.A.T))
        T.error_H1, T.error_L2  = weak_Galerkin_RT_errors(T)
        
        print("Errors: %11d %11.4e %11.4e" % (2**i, T.error_H1, T.error_L2))
        # print("Errors: %11d %11.7e %11.7e \n" % (2**i, T.error_H1, T.error_L2))

    # T.plot_solution() 
    print('------ end -------')



if __name__ == "__main__":
    main()
    # test()
    
