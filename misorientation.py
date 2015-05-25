#
#
# Some functions for computing the misorientation angle
#
#

import numpy as np
import math
import GeometryTools as GT # Used for the rotation matrix utils.


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

def compute_all_misorientation_voxel(trunc_triangles, grain_ids, M):
    """
    Voxel based computations
    """
    
    print 'Computing voxel based misorientation'
    symOps=[]
    # Unity
    symOps.append(np.eye(3))
    # 3 fold axis around z
    symOps.append(GT.rot_matrix([0,0,1],2*math.pi/3))
    symOps.append(GT.rot_matrix([0,0,-1],2*math.pi/3))
    # 2 fold axis around (sqrt(3),-1,0)
    symOps.append(GT.rot_matrix([math.sqrt(3),-1,0],math.pi))	
    # 2 fold axis around y
    symOps.append(GT.rot_matrix([0,1,0],math.pi))
    # 2 fold axis around (sqrt(3),1,0)
    symOps.append(GT.rot_matrix([math.sqrt(3),1,0],math.pi))	

    from collections import defaultdict
    areas = defaultdict(int)
    angles = defaultdict(float)
    for ix in range(M[0]):
        nx = (ix + 1) % M[0]
        for iy in range(M[1]):
            ny = (iy + 1) % M[1]
            for iz in range(M[2]):
                nz = (iz + 1) % M[2]
                ig = grain_ids[ix + iy * M[0] + iz * (M[0] * M[1])]

                if ig == 1: continue # Skip the Co-phase

                def do_compute(ng):
                    if ig != ng:
                        index = (min(ig, ng), max(ig, ng))
                        areas[index] += 1
                        if index not in angles:
                            angles[index] = compute_misorientation_net(trunc_triangles[ig-2], trunc_triangles[ng-2], symOps)

                # Check all three neighbours (in the + side)
                do_compute(grain_ids[nx + iy * M[0] + iz * (M[0] * M[1])])
                do_compute(grain_ids[ix + ny * M[0] + iz * (M[0] * M[1])])
                do_compute(grain_ids[ix + iy * M[0] + nz * (M[0] * M[1])])

    return angles, areas

#
# Updated version of compute_misorientation_net
#
def compute_misorientation_net(t1,t2,symmetry=[]):
	R1=t1.rot_matrix
	R2=t2.rot_matrix
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
# Computes misorientation given a list of trunc_trinagles and a neighborList
# Using the 001 method
#
def compute_all_misorientation_001(trunc_triangles,nbrList):

	angles=[]
	print 'Computing misorientation angles 001'
	for i,t in enumerate(trunc_triangles):
		for j in xrange(len(nbrList[i])):
		
			v1 = np.dot(t.rot_matrix,[0,0,1])
			v2 = np.dot(trunc_triangles[nbrList[i][j]].rot_matrix,[0,0,1])
			angle = np.min( [ math.acos(np.dot(v1,v2))* 180/math.pi, math.acos(np.dot(-v1,v2))* 180/math.pi ] ) 
			angles.append(angle)
	
	return angles

#
# Computes misorientation given a list of trunc_triangles and a neighborList
# Using the net rotation  M_1^-1 * M_2, with a few symmetry operations.
# This method might not be correct.
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
	
	angles=[]
	print 'Computing misorientation angles net'
	for i,t in enumerate(trunc_triangles):
		for j in xrange(len(nbrList[i])):
		
			theta =  compute_misorientation_net(t,trunc_triangles[ nbrList[i][j] ],symOps)	
			angles.append(theta)
	return angles




