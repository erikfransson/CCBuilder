# -*- coding: iso-8859-1 -*-

import sys
import salome

salome.salome_init()
theStudy = salome.myStudy

import salome_notebook
notebook = salome_notebook.NoteBook(theStudy)
sys.path.insert( 0, r'/home/sven/Backed/Microstructure/Code/CCBuilder')

###
### GEOM component
###

import GEOM
from salome.geom import geomBuilder
import math
import SALOMEDS

geompy = geomBuilder.New(theStudy)

O = geompy.MakeVertex(0, 0, 0)
OX = geompy.MakeVectorDXDYDZ(1, 0, 0)
OY = geompy.MakeVectorDXDYDZ(0, 1, 0)
OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)

geompy.addToStudy( O, 'O' )
geompy.addToStudy( OX, 'OX' )
geompy.addToStudy( OY, 'OY' )
geompy.addToStudy( OZ, 'OZ' )

import SalomeStuff as sf, numpy as np
import cPickle as pickle

#trunc_triangles_list, neighbors = sf.prepare_triangles_5(N, L)

with open('/home/sven/Backed/Microstructure/Code/CCBuilder/trunc_triangles.data', 'rb') as f:
	L = pickle.load(f)
	trunc_triangles_list = pickle.load(f)
	neighbors = pickle.load(f)

x = np.array([trunc_triangle[0].midpoint for trunc_triangle in trunc_triangles_list])
#volume = [trunc_triangle[0].volume for trunc_triangle in trunc_triangles_list]
d_eq = [trunc_triangle[0].d_eq for trunc_triangle in trunc_triangles_list]
circumcircle = [trunc_triangle[0].circumcircle for trunc_triangle in trunc_triangles_list]

sal_trunc_triangles_list = []

nr_grains = len(trunc_triangles_list)
volumes = []

volumes_no_overlap = [trunc_triangles[0].volume for trunc_triangles in trunc_triangles_list]

box = geompy.MakeBoxDXDYDZ(L, L, L)
geompy.addToStudy(box, 'Box')

total_volume = L**3
vol_frac = 0

for i, trunc_triangles in enumerate(trunc_triangles_list):
	sal_trunc_triangles_i = []
	for j, trunc_triangle_j in enumerate(trunc_triangles):
		sal_trunc_triangle_j = sf.make_truncated_triangle(geompy, trunc_triangle_j)
		
		if len(neighbors[i][j]) > 0:
			sal_trunc_triangles_nb = [sal_trunc_triangles_list[nb[0]][nb[1]] for nb in neighbors[i][j]]
			sal_trunc_triangle_j = geompy.MakeCutList(sal_trunc_triangle_j, sal_trunc_triangles_nb)
		
		sal_trunc_triangle_j = geompy.MakeCommon(sal_trunc_triangle_j, box)
		
		sal_trunc_triangles_i.append(sal_trunc_triangle_j)
	
	sal_trunc_triangles_list.append(sal_trunc_triangles_i)
	
	sal_trunc_triangle_i = geompy.MakeFuseList(sal_trunc_triangles_i)
	
	length, area, volume = geompy.BasicProperties(sal_trunc_triangle_i)
	
	geompy.addToStudy(sal_trunc_triangle_i, "TruncTriangle_" + np.str(i))
	
	vol_frac += volume / total_volume
	volumes.append(volume)
	
	print np.str(i) + " of " + np.str(len(trunc_triangles_list)) + " " + np.str(vol_frac)

if salome.sg.hasDesktop():
  salome.sg.updateObjBrowser(1)
