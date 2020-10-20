#!/usr/bin/python3
# Purpose: half-edged data structure
# Author: numanal
# Date: Jun 7, 2020
# 
# Class: 
#   BasicMesh: basic mesh with only vertices and faces
#   


# import numpy as np
# from scipy.sparse import lil_matrix
import sys





class BasicMesh(object):
    '''Basic mesh with only vertices and faces
    Attributes:
    ===========
        verts -- (Nv by 2, List, float) coordinates of vertices
        faces -- (Nt by *, List, int) vertices of each faces,
                Note: vertex indices start from 0
    Methods:
    ========
        from_obj_file(): 
            initialize from .obj file, return a mesh;
            use: T = BasicMesh.from_obj_file( fileName )
        show():
            show mesh

    '''
    def __init__(self, fileName):
        self.verts = []
        self.faces = []

        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    # vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (float(line[index1:index2]), float(line[index2:index3]) )
                    
                    self.verts.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    string = string.strip()
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            # print( string[-1], '11' )
                            face.append( int(string[i:])-1 )
                            break
                        face.append( int(string[i:string.find(" ", i)])-1 )
                        i = string.find(" ", i) + 1
                    ##
                    self.faces.append(tuple(face))

            f.close()
        except IOError:
            print(".obj file not found.")


    def show(self, faceLabelFlag=False, vertLabelFlag = False):
        '''
        '''
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection

        Nt = len(self.faces)
        fig, ax = plt.subplots()
        patches = []

        for i in range(Nt):
            n = len(self.faces[i])
            points = [self.verts[self.faces[i][j]] for j in range(n)]
    
            poly = Polygon(points, True)
            patches.append(poly)

        p = PatchCollection(patches, facecolor='white', cmap=mpl.cm.jet,\
            alpha=0.5, edgecolor='black')

        ax.add_collection(p)
        ax.autoscale_view()

        # fig.colorbar(p, ax=ax) # colorbar

        # plot elements
        if faceLabelFlag:
            for i in range(Nt):
                n = len(self.faces[i])
                iv = self.faces[i]
                xc, yc = 0., 0.
                for j in range(n):
                    xc += self.verts[iv[j]][0]
                    yc += self.verts[iv[j]][1]

                plt.text(xc/n, yc/n, str(i)+'('+str(iv[0])+')', color='r')

        # Plot points
        if vertLabelFlag:
            for i in range(len(self.verts)):
                plt.text(self.verts[i][0], self.verts[i][1], str(i), color='k')
			
        plt.title('The Basic Mesh')
        # plt.xlabel('$x$'); plt.ylabel('$y$')

        plt.show()
# --- end of BasicMesh ---



def main():

	# mesh data .obj
    fileName = "../mesh/triangles/triangle-2×4×4.obj"
    T = BasicMesh( fileName )

	# T.from_basic_mesh(basic_mesh = mesh)

    T.show(faceLabelFlag=True, vertLabelFlag=True)
	

	


if __name__ == "__main__":
	main()
