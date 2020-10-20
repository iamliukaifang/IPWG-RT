#!/usr/bin/python3
'''Purpose:
    useful class and functions
Class:
    Matrix_COO
Functions:
    area_triangle
    area_polygon
    array_to_coo
    cart2pol
    mat_mat
    dim_P
    dim_RT
    monomial_ref_element
    monomial_scaled
    vec_mat
    vec_vec

Info:
    version 0.1
'''

import numpy as np
from sys import exit
import numba as nb
from sympy.integrals.quadrature import gauss_legendre
import common

import time 


class Matrix_COO():
    '''coo mat used to save global systems
    next = 0    Start position of non-zeros elements of global stiffness matrix in COO.
                In general, Next need to be updated after putting each local matrix 
                into the global one.
    '''
    def __init__(self, nnz):
        self.nnz = nnz
        self.next = 0
        self.rows = np.zeros(nnz, dtype=common.common_int)
        self.cols = np.zeros(nnz, dtype=common.common_int)
        self.data = np.zeros(nnz, dtype=common.common_float)

    def append_submatrix(self,):
        '''
        '''
        pass



@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(nb.float64[:],nb.float64[:]), nopython = True)
def cart2pol(x, y):
    '''Purpose: Conversion between 2D coordinate systems
        Cartesian -> polar
        
    Parameters:
        x, y - (i, 1 by N) Cartesian coordinates
        rho, phi - (o, 1 by N) polar coordinates
    '''
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


# @jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(float64[:],float64[:]), nopython = True)
# def pol2cart(rho, phi):
#     '''Purpose: Conversion between 2D coordinate systems
#         polar -> Cartesian
        
#     Parameters:
#         rho, phi - (i, 1 by N) polar coordinates
#         x, y - (o, 1 by N) Cartesian coordinates
#     '''
#     x = rho * np.cos(phi)
#     y = rho * np.sin(phi)
#     return(x, y)



#------------------------------------------------------------------
# AreaTriangle:
#------------------------------------------------------------------
@nb.njit
def area_triangle(coords):
    x, y = coords[:,0], coords[:,1]
    return .5*np.absolute( (x[1]-x[0])*(y[2]-y[0])-(x[2]-x[0])*(y[1]-y[0]) )


# @nb.njit( nb.float64(nb.float64[:,:]) )
def area_polygon(coords):
    '''Purpose: compute the area of an polygon
    Parameter:
        coords	- (i, 1 by 2)   coordinates of polygonal vertices,
                                vertices list: v0, v1, ..., vn
    Date:
        Jun 4, 2020
    '''
    n = coords.shape[0]
    # new_coords = np.zeros((n+1, coords.shape[1]), dtype=nb.float64)
    new_coords = np.vstack((coords, coords[0:1,:]))
    x = new_coords[:,0]
    y = new_coords[:,1]
    return .5 * ( np.sum(x[0:n] * y[1:n+1]) - \
                np.sum(y[0:n] * x[1:n+1]) )


def polygon(coords):
    n = coords.shape[0]
    # new_coords = np.zeros((n+1, coords.shape[1]), dtype=nb.float64)
    new_coords = np.vstack((coords, coords[0:1,:]))
    x = new_coords[:,0]
    y = new_coords[:,1]
    vol = .5 * ( np.sum(x[0:n] * y[1:]) - \
                np.sum(y[0:n] * x[1:]) )
    centroid=np.zeros(2, dtype=np.float64)
    centroid[0] = np.sum((x[0:n]+x[1:])*(x[0:n]*y[1:]-x[1:]*y[0:n]))/(6*vol)
    centroid[1] = np.sum((y[0:n]+y[1:])*(x[0:n]*y[1:]-x[1:]*y[0:n]))/(6*vol)

    return vol, centroid



@nb.njit(nb.types.UniTuple(nb.float64[:],3)(nb.int64, nb.float64[:]))
def monomial_ref_element(k, coords):
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



@nb.njit(nb.types.UniTuple(nb.float64[:],3)(nb.int64, nb.float64[:], \
    nb.float64[:], nb.float64))
def monomial_scaled(k, coords, \
    coord_center=np.array([0., 0.]), h=1.):
    ''' 
    Purpose: Compute the values and derivatives scaled monomials of degree k
        m(x) = ((x-x_c)/h)^d on the (polygonal) element
    Parameters:
        k	-	(i, 1 by 1) Degree of the scaled monomial space
        coords	- (i, 1 by 2) Coordinates of (quadrature) points
        coord_center - (i, 1 by 2) coordinates of center
        h   -   (i, 1 by 1) size of the element
        V	-	(o, 1 by .5*(k+1)*(k+2)) values
        Vx	-	(o, 1 by .5*(k+1)*(k+2)) dx values
        Vy	-	(o, 1 by .5*(k+1)*(k+2)) dy values
    Date: 
        Jun 4 2020
    '''
    dim = (k+1)*(k+2)//2

    V = np.zeros(dim, dtype=np.float64)
    Vs = np.zeros(dim, dtype=np.float64)
    Vt = np.zeros(dim, dtype=np.float64)

    # monomial basis 1D
    ssn = ((coords[0]-coord_center[0])/h)**np.arange(k+1)
    ttn = ((coords[1]-coord_center[1])/h)**np.arange(k+1)

    dssn, dttn = np.zeros(k+1), np.zeros(k+1)
    dssn[1:] = np.arange(1, k+1) / h * ((coords[0]-coord_center[0])/h)**np.arange(k)
    dttn[1:] = np.arange(1, k+1) / h * ((coords[1]-coord_center[1])/h)**np.arange(k)
    ii = 0
    for i in range(k+1):  # degree i 
        for j in range(i+1): # 
            V[ii] = ssn[i-j] * ttn[j]
            Vs[ii] = dssn[i-j] * ttn[j]
            Vt[ii] = ssn[i-j] * dttn[j]
            ii += 1
	
    return V, Vs, Vt



@nb.njit(nb.int64(nb.float64[:,:], nb.int64[:],\
	nb.int64[:], nb.float64[:], nb.int64[:], nb.int64[:], nb.int64))
def array_to_coo(mat, mat_rows, mat_cols, \
                coo_data, coo_rows, coo_cols, \
                begin_position=0):
    '''
    Purpose: transform a local stiff matrix to its coo format with 
        global row and col index, and assemble it in global
        matrix in COO format.
    Parameters:
        mat      - (i, float64, M by N) The local matrix
        mat_rows - (i, int64, M by 1)
        mat_cols - (i, int64, N by 1) Global row,col index of the local matrix
        coo_data - (o, float64, M*N by 1)
        coo_rows - (o, int64, M*N by 1)
        coo_cols - (o, int64, M by 1) Global matrix in coo format
        begin_pos- (i, int64, 1 by 1) Start position to put the local matrix
    '''
    end = begin_position + mat.size
    coo_data[begin_position:end] += np.ravel(mat)
    coo_rows[begin_position:end] = np.repeat(mat_rows, mat_cols.shape[0])
    # Numba doesn't support numpy.tile, so ...
    ii = begin_position
    for i in range(mat_rows.size):
        for j in range(mat_cols.size):
            coo_cols[ii] = mat_cols[j]
            ii += 1
    # coo_cols[begin_position:end] = np.ravel(np.transpose(\
    #     np.repeat(new_cols, mat_rows.shape[0], axis=1)))

    return 0



@nb.njit(nb.float64(nb.float64[:], nb.float64[:]))
def vec_vec(A,B):
    '''
    Purpose: inner-product of two vectors
    Parameter: 
        A   -   (i, n by 1) 
        B   -   (i, n by 1) 
        C   -   (o, n by 1) C = dot (A, B)
    Last Modified:
        June 4, 2020
    '''
    C = 0.0
    for i in range( A.size ):
        C += A[i]*B[i] 
    return C


'''
Purpose: vector-matrix multiply
Parameter: 
'''
@nb.njit(nb.float64[:](nb.float64[:], nb.float64[:,:]))
def vec_mat(A,B):
    
    C = np.zeros(B.shape[1], dtype=np.float64)

    for j in range( B.shape[1] ):
        for k in range( B.shape[0] ):
            C[j] += A[k]*B[k,j]
    return C



# @nb.njit
@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]))
def mat_mat(A,B):
    '''
    Purpose: matrix multiply
    Parameter: 
    '''
    C = np.zeros((A.shape[0],B.shape[1]), dtype=np.float64)

    for i in range( A.shape[0] ):
            C[i,:] = vec_mat(A[i,:], B) 

    return C


@nb.njit
def dim_Pk(k):
    return (k+1)*(k+2)//2

@nb.njit
def dim_RT(k):
    return (k+1)*(k+3)


@nb.njit(nb.float64[:,:](nb.float64[:,:], nb.float64[:,:]))
def _mat_mat(A,B):
    '''
    Purpose: matrix multiply
    Parameter: 
    '''
    C = np.zeros((A.shape[0],B.shape[1]), dtype=np.float64)

    for i in range( A.shape[0] ):
        for j in range(0, B.shape[1]):
            for k in range(0,A.shape[1]):
                C[i,j] += A[i,k]*B[k,j] 

    return C









def test():
    a = np.ones((1000,1000), dtype=np.float64)
    c = np.zeros(a.shape, dtype=np.float64)

    s = time.time()
    # for i in range(1):a
    c = a@a
    print('Time: ', time.time()-s )
    
    s = time.time()
    # for i in range(1):
    c = _mat_mat( a,a )
    print('Time: ', time.time()-s )

    # s = time.time()
    # # for i in range(1):
    # c = mat_mat(a,a)
    # print('Time: ', time.time()-s )


def monomial_scaled_test():
    k=2
    coords = np.array([.5, .5])
    coord_center = np.array([1., 1.])
    h = 2
    t = time.time()
    m, mx, my = monomial_scaled(k, coords, coord_center, h)
    print('time elipsed: ', time.time()-t)
    print(m, mx, my)


def polygon_test():
    coords=np.array([[0.445179, 0.304621],
        [0.436864, 0.361401],
        [0.380034, 0.403298],
        [0.337488, 0.374213],
        [0.349248, 0.319155],
        [0.404193, 0.276315]])
    vol, h, centroid = polygon(coords)
    print(vol, h, centroid)


if __name__ == "__main__":
    # main()

    # polygon_test()

    # monomial_scaled_test()

    # test()

    # start = time.time()
    # test()
    # end = time.time()
    # print(end-start)

    # # test for TriangleQuadrature
    # from sympy.integrals.intpoly import *
    # import numpy as np
    # test_TriangleQuadrature()

    x = np.array([1.0, 0.0, -1.0, 0.0, 1., 1.0])
    y = np.array([+0.0, 1.0, 0.0, -1.0, -1., -0.])
    # print(y/x)
    # print(np.arctan2(y,x))

    rho, phi = cart2pol(x, y)
    # for i in range(0,rho.shape[0]):
    #     if phi[i] < 0:
    #         phi[i] += 2*np.pi
    print(rho, phi)

