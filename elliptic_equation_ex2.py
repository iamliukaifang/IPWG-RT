#!/usr/bin/python3

from numba import jit, float64
from numpy import sin, cos, pi
import numpy as np

'''IPWG parameters'''
BETA = 1
SIGMA = 400.
EPSILON = -1.0


'''Poisson's equation: 
		-Δu = f,	in Ω=[0,1]×[0,1]
	with exact solution: u = sin(pi*x) * cos(pi*x)
'''

alpha = 2.0**(-1)

@jit(float64[:](float64[:],float64[:]), nopython = True)
def u(x, y):
	return x*(x-1)*y*(y-1)*(x*x+y*y)**(.5*alpha-1)


# @jit(float64[:](float64[:],float64[:]), nopython = True)
# def ux(x, y):
# 	return pi * cos(pi*x) * sin(pi*y)


# @jit(float64[:](float64[:],float64[:]), nopython = True)
# def uy(x, y):
# 	return pi * sin(pi*x) * cos(pi*y)


@jit(float64[:](float64[:],float64[:]), nopython = True)
def f(x, y):
	return (x**2+y**2)**(.5*alpha-2)*(-alpha**2*x*y*(x*y-x-y+1) \
		-alpha*x*y*(4*x*y-2*x-2*y) \
		-2*(x**3 * (x-1)+y**3 * (y-1))+2*x*y*(4*x*y-3*(x+y)+2))





def plot_func(func, x_leftBottom=0., y_leftBottom=0., width=1., height=1.):

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Make data.
	x_step = width/50.
	y_step = height/50.
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
	plot_func(u)