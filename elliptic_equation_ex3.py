#!/usr/bin/python3

from numba import jit, float64
from numpy import sin, cos, pi
import numpy as np
import numba as nb

'''IPWG parameters'''
order = 0		#polynomial order
SIGMA = 1.
EPSILON = -1.0
BETA = 1

print("Ex 3: IPWG order=", order, "EPSILON, SIGMA, BETA = ", EPSILON, SIGMA, BETA)

'''Poisson's equation: 
		-Δu = f,	in Ω=[0,1]×[0,1]
	with exact solution: u = x*(x-1)*y*(y-1)*(x*x+y*y)**(.5*alpha-1)
'''


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
    return (rho, phi)




# alpha = 2./3.
alpha = .2

@jit(float64[:](float64[:],float64[:]))
def u(x, y):
    r, t = cart2pol(x, y)
    for i in range(t.shape[0]):
        if t[i] < 0:
            t[i] += 2 * pi
    return r**alpha * sin(alpha * t)


@jit(float64[:](float64[:],float64[:]), nopython = True)
def f(x, y):
	return np.zeros(x.shape, dtype=np.float64)
	# return u(x, y)





def plot_func(func, x_leftBottom=0., y_leftBottom=0., width=1., height=1.):

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Make data.
	x_step = width/80.
	y_step = height/80.
	X = np.arange(x_leftBottom, x_leftBottom+width+x_step, x_step)
	Y = np.arange(y_leftBottom, y_leftBottom+height+y_step, y_step)
	X, Y = np.meshgrid(X, Y)

	Z = np.zeros(X.shape)
	for j in range(X.shape[0]):
		Z[j,:] = func(X[j,:], Y[j,:])

	# Plot the surface.
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
						linewidth=0, antialiased=False)

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()



if __name__ == "__main__":

	'''
	Test: plot_func()
	'''
	# This import registers the 3D projection, but is otherwise unused.
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

	import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter

	# plot_func(ux, -1., -1., 2., 2.)
	plot_func(u, -1., -1., 2., 2.)