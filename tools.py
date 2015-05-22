#
#
# Some scripts for computing the misorientation angle
#
#

import numpy as np
import math
import GeometryTools as GT

#
# Finds a neighborList using circumcircle similar to how I think Sven tried to implement it.
#
def findNeighbors(trunc_triangles,L):

	print 'Computing neighbor list'
	nbrList=[]
	for i,ti in enumerate(trunc_triangles):
		nb=[]	
		copies_i = []
		copies_i.append(ti)
		copies_i.extend(ti.find_periodic_copies(L))
		for j,tj in enumerate(trunc_triangles):
			ij_nb = False	
			copies_j = []
			copies_j.append(tj)
			copies_j.extend(tj.find_periodic_copies(L))
		
			if ( i!=j ):
			
				for ic,i_copy in enumerate(copies_i):
					for jc,j_copy in enumerate(copies_j):
						if ( np.linalg.norm(i_copy.midpoint - j_copy.midpoint) < i_copy.circumcircle + j_copy.circumcircle):
							ij_nb=True
							#print i,j
	
			if ij_nb:
				nb.append(j)
		nbrList.append(np.array(nb))


	return nbrList

#
# Find a neighborList using voxels.
#
def find_Neighbors_voxel():
	print 'Implement me!'






#
# Computes misorientation given a list of trunc_trinagles and a neighborList
# Using the 001 method
#
def compute_all_misorientation_001(trunc_triangles,nbrList):

	angles=[]
	print 'Computing misorientation angles'
	for i,t in enumerate(trunc_triangles):
		for j in xrange(len(nbrList[i])):
		
			v1 = np.dot(t.rot_matrix,[0,0,1])
			v2 = np.dot(trunc_triangles[nbrList[i][j]].rot_matrix,[0,0,1])
			angle = np.min( [ math.acos(np.dot(v1,v2))* 180/math.pi, math.acos(np.dot(-v1,v2))* 180/math.pi ] ) 
			angles.append(angle)
	
	return angles
#
# Computes misorientation given a list of trunc_triangles and a neighborList
# Using the net rotation  M_1^-1 * M_2
#
def compute_all_misorientation_net(trunc_triangles,nbrList):
	symOps=[]
	# Unity
	symOps.append(np.eye(3))
	# 3 fold axis around z
	symOps.append(GT.rot_matrix([0,0,1],2*math.pi/3))
	symOps.append(GT.rot_matrix([0,0,-1],2*math.pi/3))
	# 2 fold axis around y
	symOps.append(GT.rot_matrix([0,1,0],math.pi))
	symOps=[]
	angles=[]
	print 'Computing misorientation angles'
	for i,t in enumerate(trunc_triangles):
		for j in xrange(len(nbrList[i])):
		
			theta =  compute_misorientation_net(t,trunc_triangles[ nbrList[i][j] ],symOps)	
			angles.append(theta)
	return angles



#
# One way of computing misorientation angle between two grains t1 t2.
# Looking at how the [0,0,1] direction is changed
#
def compute_misorientation_001(t1,t2):
	v1 = np.dot(t1.rot_matrix,[0,0,1])
	v2 = np.dot(t2.rot_matrix,[0,0,1])
	angle = math.acos(np.dot(v1,v2))* 180/math.pi
	if angle > 90:
		angle = 180-angle
	return angle

#
# One way of computing misorientation angle between two grains t1 t2.
# Using  Net rotation = g_b * g_a ^(-1)
# Considering symmetries.
def compute_misorientation_net(t1,t2,symmetry=[]):
	rot1 = t1.rot_matrix
	rot2 = t2.rot_matrix
	rot2_inv = np.linalg.inv(t2.rot_matrix)
	
	net_rotation=np.dot(rot1,rot2_inv)
	
	if len(symmetry)==0:	
		theta,axis = GT.rotmatrix_to_axisangle(net_rotation)
		return theta*180/math.pi
	else:
		angles=[]
		for i in xrange(len(symmetry)):
			for j in xrange(len(symmetry)):	
				theta,axis = GT.rotmatrix_to_axisangle( np.dot( np.dot(rot1,symmetry[i]),np.dot(rot2_inv,symmetry[j]) )	)
				angles.append(theta*180/math.pi)
		
		return np.min(angles)

#
# Test function for misorientation, given two rotation matrices R1 and R2
#
def test_misorientation(R1,R2,symmetry=[]):
	R2inv = np.linalg.inv(R2)
	net_rotation=np.dot(R1,R2inv)
	
	if len(symmetry)==0:	
		theta,axis = GT.rotmatrix_to_axisangle(net_rotation)
		return theta*180/math.pi
	else:
		angles=[]
		for i in xrange(len(symmetry)):
			for j in xrange(len(symmetry)):	
				theta,axis = GT.rotmatrix_to_axisangle( np.dot( np.dot(symmetry[i],R1),np.dot(R2inv,symmetry[j]) )	)
				angles.append(theta*180/math.pi)
		
		return np.min(angles)




#
# Plot vertices
#
def plot_vertices(ax,vertices,t):
		
	x0=vertices[:,0]
	x0=np.append(x0,vertices[0][0])
	x0=np.append(x0,x0)
	y0=vertices[:,1]
	y0=np.append(y0,vertices[0][1])
	y0=np.append(y0,y0)
	z0=vertices[:,2]
	z0=np.append(z0,vertices[0][2])
	z0=np.append(z0,-z0)
	ax.plot(x0,y0,z0,'b')
	for i in range(6):
		ax.plot([x0[i],x0[i]],[y0[i],y0[i]],[z0[i],z0[i]+t],'b')
	
#
# Plot real vertices
#
def plot_vertices_real(ax,vertices_real):
	x1 = []
	y1 = []
	z1 = []
	x2 = []
	y2 = []
	z2 = []
	for i in range(6):
		x1.append(vertices_real[i*2][0])
		y1.append(vertices_real[i*2][1])
		z1.append(vertices_real[i*2][2])
		x2.append(vertices_real[i*2+1][0])
		y2.append(vertices_real[i*2+1][1])
		z2.append(vertices_real[i*2+1][2])

	x1.append(vertices_real[0][0])
	y1.append(vertices_real[0][1])
	z1.append(vertices_real[0][2])

	x2.append(vertices_real[1][0])
	y2.append(vertices_real[1][1])
	z2.append(vertices_real[1][2])
	
	
	ax.plot(x1,y1,z1,'b')	
	ax.plot(x2,y2,z2,'b')
	for i in range(6):
		ax.plot([x1[i],x2[i]],[y1[i],y2[i]],[z1[i],z2[i]],'b')
	




