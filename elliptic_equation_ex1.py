#!/usr/bin/python3

from numba import jit, float64
from numpy import sin, cos, pi
import numpy as np


'''Poisson's equation: 
		-Δu = f,	in Ω=[0,1]×[0,1]
	with exact solution: u = sin(k*pi*x) * cos(k*pi*x)
'''

k = .5

@jit(float64[:](float64[:],float64[:]), nopython = True)
def u(x, y):
	return sin(k*pi*x) * sin(k*pi*y)


@jit(float64[:](float64[:],float64[:]), nopython = True)
def ux(x, y):
	return k*pi * cos(k*pi*x) * sin(k*pi*y)


@jit(float64[:](float64[:],float64[:]), nopython = True)
def uy(x, y):
	return k*pi * sin(k*pi*x) * cos(k*pi*y)


@jit(float64[:](float64[:],float64[:]), nopython = True)
def f(x, y):
	return 2*(k*pi)**2 * sin(k*pi*x) * sin(k*pi*y)





def plot_func(func, x_leftBottom=0., y_leftBottom=0., width=1., height=1., N = 50):

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# Make data.
	x_step = width/N
	y_step = height/N
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