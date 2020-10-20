#!/usr/bin/python3
# Purpose: finite element mesh data structure
# 
# Class: 
#   PolygonMesh: 
#   todo: 
# 


import numpy as np
from scipy.sparse import lil_matrix
import sys

from basicmesh import BasicMesh


class PolygonMesh(object):
	''' Polygon mesh  (without hangling nodes)
	Attributes:
    ===========
		Nt: the number of polygons
		Ne: the number of edges
		Nv: the number of nodes

		face_edges:  (Nt by *) 
				the edges boadering the face, in counterclockwise order.
		face_edge_num: (Nt by 1, int) number of edges of each face
		face_edge_orientations: (Nt by *, boolean) 
				It indicates the orientations of edges. if True, orientation
				is clockwise from the first endpoint to the second.
				Otherwise, orientation is counterclockwise from the second
				endpoint to the first.
		edge_verts: (Ne by 2, int) Pointers of two endpoints.
		edge_faces: (Ne by 2, int) Two adjacent elements.
				But if the edge is not interior, its second position is -1.
		edge_local_indices: (Ne by 2, int) Two local indices in the adjacent faces.
				But if the edge is not interior, its second position is -1.
		verts: (Nv by 2, array). Coordinates of nodes.
	Methods:
    ========
		from_basic_mesh(): 
	'''
	def __init__(self, basic_mesh ):
		'''Init from .obj file
		'''

        # Purpose of the private variables: 
        #   compute them when they are really needed
        # Note: set to None after any change of the previous data
		self.__face_areas = None
		self.__face_sizes = None
		self.__face_verts = None
		self.__face_centroids = None

		
		'''Transform from the basic mesh, which only contains vertices and faces
		'''
		Nv, Nt = len(basic_mesh.verts), len(basic_mesh.faces)
		# - estermated upper bound of the number of edges by Euler formula
		Ne_est = (Nv + (Nt + 1) - 2 )

		self.edge_verts = np.zeros((Ne_est,2), dtype=np.int) - 1
		self.edge_faces = np.zeros((Ne_est,2), dtype=np.int) - 1
		self.edge_local_indices = np.zeros((Ne_est,2), dtype=np.int) - 1
		self.verts = np.array( basic_mesh.verts, dtype=np.float )

		self.face_edge_num = np.zeros(Nt, dtype=np.int)
		# - edge number of each face
		for i in range(Nt):
			self.face_edge_num[i] = len(basic_mesh.faces[i])

		self.face_edges = np.zeros((Nt, np.amax(self.face_edge_num)), dtype=np.int) - 1 
		self.face_edge_orientations = np.zeros((Nt, np.amax(self.face_edge_num)), dtype=np.bool)

		# Loop over all polygon
		# - number of edges starts from 0 (i.e. (Ne+1)th edge)
		Ne = 0
		# - vertices connection matrix, recording edges found
		vcm = lil_matrix((Nv,Nv), dtype=np.int) 

		for i in range(Nt):
			verts = np.array(basic_mesh.faces[i])  # vertices of the face
			n_edges = verts.shape[0] # number of edges of this face

			verts2 = np.append(verts, verts[0])
			# loop over all edges
			for j in range(n_edges):
				v1, v2 = verts2[j], verts2[j+1]
				# 0 (new edge), or old edge
				if vcm[v1,v2] == 0:

					vcm[v1,v2] = Ne+1
					vcm[v2,v1] = Ne+1
					self.face_edges[i,j] = Ne
					self.face_edge_orientations[i,j] = True
					self.edge_verts[Ne,0] = v1
					self.edge_verts[Ne,1] = v2
					self.edge_faces[Ne,0] = i
					self.edge_local_indices[Ne,0] = j

					Ne += 1 # new edge
				else:
					self.face_edges[i,j] = vcm[v1,v2]-1
					# T.face_edge_orientations[i,j] = False
					self.edge_faces[vcm[v1,v2]-1, 1] = i 
					self.edge_local_indices[vcm[v1,v2]-1, 1] = j

		self.edge_faces = self.edge_faces[:Ne,:]
		self.edge_verts = self.edge_verts[:Ne,:]
		self.edge_local_indices = self.edge_local_indices[:Ne,:]


	# @staticmethod
	# def from_basic_mesh( basic_mesh ):
	# 	'''Transform from the basic mesh, which only contains vertices and faces
	# 	'''
	# 	# Initialize
	# 	T = PolygonMesh()

	# 	Nv, Nt = len(basic_mesh.verts), len(basic_mesh.faces)
	# 	# - estermated upper bound of the number of edges by Euler formula
	# 	Ne_est = (Nv + (Nt + 1) - 2 )

	# 	T.edge_verts = np.zeros((Ne_est,2), dtype=np.int) - 1
	# 	T.edge_faces = np.zeros((Ne_est,2), dtype=np.int) - 1
	# 	T.edge_local_indices = np.zeros((Ne_est,2), dtype=np.int) - 1
	# 	T.verts = np.array( basic_mesh.verts, dtype=np.float )

	# 	T.face_edge_num = np.zeros(Nt, dtype=np.int)
	# 	# - edge number of each face
	# 	for i in range(Nt):
	# 		T.face_edge_num[i] = len(basic_mesh.faces[i])

	# 	T.face_edges = np.zeros((Nt, np.amax(T.face_edge_num)), dtype=np.int) - 1 
	# 	T.face_edge_orientations = np.zeros((Nt, np.amax(T.face_edge_num)), dtype=np.bool)

	# 	# Loop over all polygon
	# 	# - number of edges starts from 0 (i.e. (Ne+1)th edge)
	# 	Ne = 0
	# 	# - vertices connection matrix, recording edges found
	# 	vcm = lil_matrix((Nv,Nv), dtype=np.int) 

	# 	for i in range(Nt):
	# 		verts = np.array(basic_mesh.faces[i])  # vertices of the face
	# 		n_edges = verts.shape[0] # number of edges of this face

	# 		verts2 = np.append(verts, verts[0])
	# 		# loop over all edges
	# 		for j in range(n_edges):
	# 			v1, v2 = verts2[j], verts2[j+1]
	# 			# 0 (new edge), or old edge
	# 			if vcm[v1,v2] == 0:

	# 				vcm[v1,v2] = Ne+1
	# 				vcm[v2,v1] = Ne+1
	# 				T.face_edges[i,j] = Ne
	# 				T.face_edge_orientations[i,j] = True
	# 				T.edge_verts[Ne,0] = v1
	# 				T.edge_verts[Ne,1] = v2
	# 				T.edge_faces[Ne,0] = i
	# 				T.edge_local_indices[Ne,0] = j

	# 				Ne += 1 # new edge
	# 			else:
	# 				T.face_edges[i,j] = vcm[v1,v2]-1
	# 				# T.face_edge_orientations[i,j] = False
	# 				T.edge_faces[vcm[v1,v2]-1, 1] = i 
	# 				T.edge_local_indices[vcm[v1,v2]-1, 1] = j

	# 	T.edge_faces = T.edge_faces[:Ne,:]
	# 	T.edge_verts = T.edge_verts[:Ne,:]
	# 	T.edge_local_indices = T.edge_local_indices[:Ne,:]

	# 	return T


	# @staticmethod
	# def from_obj_file( fileName ):
	# 	'''Load data from obj file
	# 	'''
	# 	# Load basic mesh consisting of vertices and faces
	# 	mesh = BasicMesh.from_obj_file( fileName )

	# 	T = PolygonMesh.from_basic_mesh(mesh)

	# 	return T

    
	# Compute __face_verts
	def _update_face_vertices(self):
		self.__face_verts = np.zeros(self.face_edges.shape, dtype=np.int) - 1
		Nt = self.face_edges.shape[0]
		for i in range(Nt):
			nf = self.face_edge_num[i]
			for j in range(nf):
				if self.face_edge_orientations[i,j]:
					self.__face_verts[i, j] = self.edge_verts[self.face_edges[i,j],0]
				else:
					self.__face_verts[i, j] = self.edge_verts[self.face_edges[i,j],1]


	@property
	def face_verts(self):
		if self.__face_verts is None:
			self._update_face_vertices()
		return self.__face_verts


	# Compute __face_areas
	def _update_face_areas(self):
		Nt = self.face_edges.shape[0]
		self.__face_areas = np.zeros(Nt, dtype=np.float)
		V = np.zeros(self.face_edges.shape[1]+1, dtype=np.int) - 1 
		for i in range(Nt):
			nf = self.face_edge_num[i]
			# for j in range(nf):
			# 	if self.face_edge_orientations[i,j]:
			# 		V[j] = self.edge_verts[self.face_edges[i,j],0]
			# 	else:
			# 		V[j] = self.edge_verts[self.face_edges[i,j],1]
			# V[nf] = V[0]
			# coords = self.verts[V[0:nf+1],:]
			V[:nf] = self.face_verts[i,:nf]
			V[nf] = V[0]
			coords = self.verts[V,:]

			self.__face_areas[i] = .5 * ( np.sum(coords[0:nf,0] * coords[1:nf+1, 1]) - \
				np.sum(coords[0:nf,1] * coords[1:nf+1, 0]) )


	@property
	def face_areas(self):
		if self.__face_areas is None:
			self._update_face_areas()
		return self.__face_areas

    
	# Compute __face_sizes: diameter of circumcircle
	def _update_face_sizes(self):
		Nt = self.face_edges.shape[0]
		self.__face_sizes = np.zeros(Nt, dtype=np.float)

		for i in range(Nt):
			nf = self.face_edge_num[i]
			iv = self.face_verts[i,:nf]
			coords = self.verts[iv,:]
			# print(coords,'l')
			center = np.sum(coords, axis=0) / nf
			self.__face_sizes[i] = 2 * np.amax(np.linalg.norm(coords - center, axis=1))


	@property
	def face_sizes(self):
		if self.__face_sizes is None:
			self._update_face_sizes()
		return self.__face_sizes

    
	def _update_face_centroids(self):
		Nt = self.face_edges.shape[0]
		self.__face_centroids = np.zeros((Nt, 2), dtype=np.float)
		for i in range(Nt):
			n = self.face_edge_num[i]
			iv = self.face_verts[i,:n]
			self.verts[iv,:]
			new_coords = np.vstack((self.verts[iv,:], self.verts[iv[0],:]))
			x = new_coords[:,0]
			y = new_coords[:,1]
			centroid = np.zeros(2, dtype=np.float64)
			centroid[0] = np.sum((x[:n]+x[1:])*(x[:n]*y[1:]-x[1:]*y[:n]))/(6*self.face_areas[i])
			centroid[1] = np.sum((y[:n]+y[1:])*(x[:n]*y[1:]-x[1:]*y[:n]))/(6*self.face_areas[i])
			self.__face_centroids[i,:] = centroid


	@property
	def face_centroids(self):
		if self.__face_centroids is None:
			self._update_face_centroids()
		return self.__face_centroids


	def get_face_verts(self, i):
		n = self.face_edge_num[i]
		edges = self.face_edges[i,:n]
		os = self.face_edge_orientations[i,:]
		vs = np.zeros(n, dtype=np.int)
		for k in range(n):
			if os[k]:
				vs[k] = self.edge_verts[edges[k],0]
			else:
				vs[k] = self.edge_verts[edges[k],1]
		
		return vs


	def show(self, face_label_flag=False, edge_label_flag = False, vert_label_flag = False, \
		mesh_title=None, axis=True):
		'''Mesh show
		'''
		import matplotlib.pyplot as plt
		import matplotlib as mpl
		from matplotlib.patches import Polygon
		from matplotlib.collections import PatchCollection

		Nt = self.face_edges.shape[0]
		fig, ax = plt.subplots()
		patches = []

		for i in range(Nt):
			n = self.face_edge_num[i]
			iv = self.get_face_verts(i)
			poly = Polygon(self.verts[iv,:], True)
			patches.append(poly)

		p = PatchCollection(patches, facecolor='white', cmap=mpl.cm.jet,\
            alpha=0.5, edgecolor='black')

		ax.add_collection(p) 
		ax.autoscale_view()

		# fig.colorbar(p, ax=ax) # colorbar

		# plot elements
		if face_label_flag:
			for i in range(Nt):
				n = self.face_edge_num[i]
				iv = self.get_face_verts(i)
				coords = self.verts[iv,:]
				c = np.sum(coords, axis=0)/n
				plt.text(c[0],c[1], str(i)+'('+str(iv[0])+')', color='r')

		# plot edges
		if edge_label_flag:
			for i in range(0,self.edge_verts.shape[0]):
				iv = self.edge_verts[i,:]
				m = 0.5*np.sum(self.verts[iv,:], axis=0)
				plt.text(m[0], m[1]+.01, str(i)+'('+str(iv[0])+')', color='g')

		# Plot points
		if vert_label_flag:
			for i in range(0,self.verts.shape[0]):
				plt.text(self.verts[i,0], self.verts[i,1], str(i), color='k')
		if mesh_title is not None:
			plt.title(mesh_title)
        # plt.xlabel('$x$'); plt.ylabel('$y$')
		if not axis:
			plt.axis('off')

		plt.show()


	def show_with_turtle(self, speed=5):
		'''Mesh show with turtle
		'''
		import turtle 

		V = self.verts.copy()

		# map to [-.5, -.5]*[.5, .5]
		V[:,0] /= np.absolute(np.amax(V[:,0]))
		V[:,1] /= np.absolute(np.amax(V[:,1]))
		V = (self.verts - .5)*500

		turtle.title('Turtle')
		turtle.speed( speed ) # set speed
		turtle.color('#ff9980', '#ffffb3')
		turtle.penup()
		for i in range(self.face_edges.shape[0]):
			n = self.face_edge_num[i]
			iv = self.get_face_verts(i)
			turtle.goto( V[iv[0],:] )
			turtle.begin_fill()
			turtle.pendown()
			for j in range(1,n):
				turtle.goto( V[iv[j],:])

			turtle.goto( V[iv[0],:])
			turtle.end_fill()

			turtle.penup()
		
		turtle.done()

# --- end of PolygonMesh ---


def main():

	# mesh data .obj
	fileName = "./mesh/triangles/triangle-2x2x2.obj"

	# Load basic mesh consisting of vertices and faces
	basic_mesh = BasicMesh( fileName )

	T = PolygonMesh(basic_mesh)
	# T.verts = 2*T.verts-1
	T.show()


	print('End.')

	


if __name__ == "__main__":

	from scipy.spatial import Delaunay
	import triangle as tr
	import matplotlib.pyplot as plt

	main()
