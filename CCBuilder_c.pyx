# cython: boundscheck=False, wraparound=False, profile=False, cdivision=True

import cython
import numpy as np
cimport numpy as np
#from cpython cimport bool
import TruncatedTriangle

cdef extern from "math.h":
	double sqrt(double x)

cdef extern from "math.h":
	double floor(double x)

cdef inline int int_max(int a, int b):
	return a if a >= b else b

cdef inline int int_min(int a, int b):
	return a if a <= b else b

cdef inline double mod(double x1, double x2):
	return x1 - floor(x1 / x2) * x2

# Assume that i2 > 0
# C %-operator is retarded
cdef inline int int_mod(int i1, int i2):
	return i1%i2 if i1 >= 0 else ((i1%i2) + i2) % i2

# To convince myself that mod above is correct and that % is no good
#def modulo1(x1, x2):
#	cdef double dx1, dx2
#	dx1 = <double> x1
#	dx2 = <double> x2
#	return dx1 % dx2

#def modulo2(x1, x2):
#	return mod(x1, x2)

#def int_modulo(i1, i2):
#	cdef int ii1, ii2, r
#	ii1 = <int> i1
#	ii2 = <int> i2
#	r = int_mod(ii1, ii2)
#	return r

#def int_modulo2(i1, i2):
#	cdef int ii1, ii2, r
#	ii1 = <int> i1
#	ii2 = <int> i2
#	r = ii1 % ii2
#	return r

#cdef inline double square(double x):
#	return x*x

#@cython.profile(False)
#def sum_potential3(np.ndarray[double, ndim=1, mode="c"] r not None, double L, np.ndarray[double, ndim=1, mode="c"] circumcircle not None, np.ndarray[double, ndim=1, mode="c"] volume not None):
#	cdef int N, i, j, ix, iy, iz
#	cdef double U, r_ix, r_iy, r_iz, r_jx, r_jy, r_jz, r_0, r_ijx, r_ijy, r_ijz, r_ij_norm
	
#	U = 0.0
#	N = r.shape[0] / 3
	
#	for i in range(N):
#		r_ix = r[3*i] % L
#		r_iy = r[3*i+1] % L
#		r_iz = r[3*i+2] % L
#		for j in range(i+1, N):
#			r_jx = r[3*j] % L
#			r_jy = r[3*j+1] % L
#			r_jz = r[3*j+2] % L
#			r_0 = circumcircle[i] + circumcircle[j]
#			for ix in range(-1,2):
#				r_ijx = r_jx + ix*L - r_ix
#				for iy in range(-1,2):
#					r_ijy = r_jy + iy*L - r_iy
#					for iz in range(-1,2):
#						r_ijz = r_jz + iz*L - r_iz
#						r_ij_norm = sqrt(square(r_ijx) + square(r_ijy) + square(r_ijz))
#						if r_ij_norm < r_0:
#							U += volume[i]*volume[j]*square((r_ij_norm - r_0) / r_0)
	
#	return U

#@cython.profile(False)
#def sum_potential3_grad(np.ndarray[double, ndim=1, mode="c"] r not None, double L, np.ndarray[double, ndim=1, mode="c"] circumcircle not None, np.ndarray[double, ndim=1, mode="c"] volume not None):
#	cdef int N, i, j, ix, iy, iz
#	cdef double r_ix, r_iy, r_iz, r_jx, r_jy, r_jz, r_0, r_ijx, r_ijy, r_ijz, r_ij_norm
	
#	N = r.shape[0] / 3
	
#	cdef np.ndarray[double, ndim=1] U_grad = np.zeros(dtype="d", shape=(r.shape[0]))
	
#	for i in range(N):
#		r_ix = r[3*i] % L
#		r_iy = r[3*i+1] % L
#		r_iz = r[3*i+2] % L
#		for j in range(i+1, N):
#			r_jx = r[3*j] % L
#			r_jy = r[3*j+1] % L
#			r_jz = r[3*j+2] % L
#			r_0 = circumcircle[i] + circumcircle[j]
#			for ix in range(-1,2):
#				r_ijx = r_jx + ix*L - r_ix
#				for iy in range(-1,2):
#					r_ijy = r_jy + iy*L - r_iy
#					for iz in range(-1,2):
#						r_ijz = r_jz + iz*L - r_iz
#						r_ij_norm = sqrt(square(r_ijx) + square(r_ijy) + square(r_ijz))
#						if r_ij_norm < r_0:
#							grad = volume[i]*volume[j]*2/square(r_0) * (1 - r_0/r_ij_norm)
#							U_grad[3*i] -= grad*r_ijx
#							U_grad[3*i+1] -= grad*r_ijy
#							U_grad[3*i+2] -= grad*r_ijz
#							U_grad[3*j] += grad*r_ijx
#							U_grad[3*j+1] += grad*r_ijy
#							U_grad[3*j+2] += grad*r_ijz
	
#	return U_grad

def sum_potential3_and_grad(np.ndarray[double, ndim=1, mode="c"] r not None, double L, np.ndarray[double, ndim=1, mode="c"] circumcircle not None, np.ndarray[double, ndim=1, mode="c"] volume not None):
	cdef int N, i, j, ix, iy, iz
	cdef double U, r_ix, r_iy, r_iz, r_jx, r_jy, r_jz, r_0, r_ijx, r_ijy, r_ijz, r_ij_norm, grad, sq
	
	N = r.shape[0] / 3
	
	U = 0.0
	cdef np.ndarray[double, ndim=1] U_grad = np.zeros(dtype="d", shape=(r.shape[0]))
	
	for i in range(N):
		r_ix = mod(r[3*i], L)
		r_iy = mod(r[3*i+1], L)
		r_iz = mod(r[3*i+2], L)
		for j in range(i+1, N):
			r_jx = mod(r[3*j], L)
			r_jy = mod(r[3*j+1], L)
			r_jz = mod(r[3*j+2], L)
			r_0 = circumcircle[i] + circumcircle[j]
			for ix in range(-1,2):
				r_ijx = r_jx + ix*L - r_ix
				for iy in range(-1,2):
					r_ijy = r_jy + iy*L - r_iy
					for iz in range(-1,2):
						r_ijz = r_jz + iz*L - r_iz
						r_ij_norm = sqrt(r_ijx*r_ijx + r_ijy*r_ijy + r_ijz*r_ijz)
						if r_ij_norm < r_0:
							sq = (r_ij_norm - r_0) / r_0
							U += volume[i]*volume[j]*sq*sq
							grad = volume[i]*volume[j]*2/(r_0*r_0) * (1 - r_0/r_ij_norm)
							U_grad[3*i] -= grad*r_ijx
							U_grad[3*i+1] -= grad*r_ijy
							U_grad[3*i+2] -= grad*r_ijz
							U_grad[3*j] += grad*r_ijx
							U_grad[3*j+1] += grad*r_ijy
							U_grad[3*j+2] += grad*r_ijz
	
	return (U, U_grad)

def populate_voxels(double L, int M, trunc_triangles_list):
	print "Populating voxels"
	
	cdef int M3, M2, i, j, min_ix, max_ix, min_iy, max_iy, min_iz, max_iz, ix, iy, iz, N, N_vert, index
	cdef double delta_x
	
	M3 = M*M*M
	M2 = M*M
	
	cdef np.ndarray[int, ndim=1] grain_ids = np.ones(dtype="int32", shape=(M3))
	cdef np.ndarray[int, ndim=1] phases = np.ones(dtype="int32", shape=(M3))
	cdef np.ndarray[unsigned char, ndim=1] good_voxels = np.ones(dtype="uint8", shape=(M3))
	cdef np.ndarray[float, ndim=2] euler_angles = np.zeros(dtype="float32", shape=(M3, 3))
	
	cdef np.ndarray[int, ndim=1] phase_volumes = np.zeros(dtype="int32", shape=(2))
	phase_volumes[0] = M3
	phase_volumes[1] = 0
	
	delta_x = L/M
	
	N = len(trunc_triangles_list)
	cdef np.ndarray[int, ndim=1] grain_volumes = np.zeros(dtype="int32", shape=(N))
	
	# Needed to calculate if inside
	cdef np.ndarray[double, ndim=2, mode="c"] vert
	cdef np.ndarray[double, ndim=1, mode="c"] euler_angles_j
	cdef np.ndarray[double, ndim=1, mode="c"] midpoint
	cdef np.ndarray[double, ndim=2, mode="c"] rot_matrix_tr
	cdef np.ndarray[double, ndim=1] r0 = np.zeros(dtype="float64", shape=(3))
	cdef np.ndarray[double, ndim=1] r1 = np.zeros(dtype="float64", shape=(3))
	
	cdef double t
	cdef long truncation, inside
	
	for i, trunc_triangles in enumerate(trunc_triangles_list):
		#print "Grain " + str(i) + " of " + str(N)
		for j, tr_tri in enumerate(trunc_triangles):
			vert = tr_tri.vertices
			N_vert = vert.shape[0]
			
			euler_angles_j = tr_tri.euler_angles
			midpoint = tr_tri.midpoint
			rot_matrix_tr = tr_tri.rot_matrix_tr
			t = tr_tri.t
			truncation = tr_tri.r > 0
			
			min_ix = int_max(0, <int> floor(M*tr_tri.min_x/L))
			max_ix = int_min(M-1, <int> floor(M*tr_tri.max_x/L))
			min_iy = int_max(0, <int> floor(M*tr_tri.min_y/L))
			max_iy = int_min(M-1, <int> floor(M*tr_tri.max_y/L))
			min_iz = int_max(0, <int> floor(M*tr_tri.min_z/L))
			max_iz = int_min(M-1, <int> floor(M*tr_tri.max_z/L))
			
			# grain ids: binder 1, WC 2+i where i=0,...
			# phases: binder 1, WC 2
			# All code below should translate to good C
			for iz in range(min_iz, max_iz+1):
				for iy in range(min_iy, max_iy+1):
					for ix in range(min_ix, max_ix+1):
						index = ix + iy*M + iz*M2
						if grain_ids[index] == 1: # still unclaimed binder
							r0[0] = delta_x*(0.5+ix) - midpoint[0]
							r0[1] = delta_x*(0.5+iy) - midpoint[1]
							r0[2] = delta_x*(0.5+iz) - midpoint[2]
							
							# Rotate to coordinates of the triangle
							# Use explicit matrix mult to avoid calling numpy
							r1[0] = rot_matrix_tr[0,0]*r0[0] + rot_matrix_tr[0,1]*r0[1] + rot_matrix_tr[0,2]*r0[2]
							r1[1] = rot_matrix_tr[1,0]*r0[0] + rot_matrix_tr[1,1]*r0[1] + rot_matrix_tr[1,2]*r0[2]
							r1[2] = rot_matrix_tr[2,0]*r0[0] + rot_matrix_tr[2,1]*r0[1] + rot_matrix_tr[2,2]*r0[2]
							
							# The triangle is within the x-y plane. Check if r1 is inside the truncated triangle
							if truncation:
								inside = (r1[2] > -t*0.5 and r1[2] < t*0.5 and
								(r1[1] - vert[2,1])*(vert[1,0] - vert[2,0]) - (vert[1,1] - vert[2,1])*(r1[0] - vert[2,0]) < 0.0 and
								(r1[1] - vert[3,1])*(vert[2,0] - vert[3,0]) - (vert[2,1] - vert[3,1])*(r1[0] - vert[3,0]) < 0.0 and
								(r1[1] - vert[4,1])*(vert[3,0] - vert[4,0]) - (vert[3,1] - vert[4,1])*(r1[0] - vert[4,0]) < 0.0 and
								(r1[1] - vert[4,1])*(vert[5,0] - vert[4,0]) - (vert[5,1] - vert[4,1])*(r1[0] - vert[4,0]) > 0.0 and
								(r1[1] - vert[5,1])*(vert[0,0] - vert[5,0]) - (vert[0,1] - vert[5,1])*(r1[0] - vert[5,0]) > 0.0 and
								(r1[1] - vert[0,1])*(vert[1,0] - vert[0,0]) - (vert[1,1] - vert[0,1])*(r1[0] - vert[0,0]) > 0.0)
							else:
								inside = (r1[2] > -t*0.5 and r1[2] < t*0.5 and
								(r1[1] - vert[1,1])*(vert[0,0] - vert[1,0]) - (vert[0,1] - vert[1,1])*(r1[0] - vert[1,0]) < 0.0 and
								(r1[1] - vert[2,1])*(vert[1,0] - vert[2,0]) - (vert[1,1] - vert[2,1])*(r1[0] - vert[2,0]) < 0.0 and
								(r1[1] - vert[2,1])*(vert[0,0] - vert[2,0]) - (vert[0,1] - vert[2,1])*(r1[0] - vert[2,0]) > 0.0)
							
							if inside:
								grain_ids[index] = i+2
								phases[index] = 2
								euler_angles[index, 0] = <float> euler_angles_j[0]
								euler_angles[index, 1] = <float> euler_angles_j[1]
								euler_angles[index, 2] = <float> euler_angles_j[2]
								phase_volumes[0] -= 1
								phase_volumes[1] += 1
								grain_volumes[i] += 1
	
	return grain_ids, phases, good_voxels, euler_angles, phase_volumes, grain_volumes

def calc_surface_prop(int M, np.ndarray[int, ndim=1] grain_ids):
	print "Calculating surface properties"
	
	cdef int M2, M3, iz, iy, ix, i, grain_id, nb_id
	cdef int nb_indices[6]
	M2 = M*M
	M3 = M2*M
	
	cdef np.ndarray[char, ndim=1] surface_voxels = np.zeros(M3, dtype='int8')
	cdef np.ndarray[char, ndim=1] gb_voxels = np.zeros(M3, dtype='int8')
	cdef np.ndarray[char, ndim=1] interface_voxels = np.zeros(M3, dtype='int8')
	
	# Calculate surface voxels for all phases consistently with Dream3D, except that Dream3D does not seem to use periodic boundaries.
	# Interface and grain boundary voxels are only calculated for the WC phase.
	for iz in range(M):
		for iy in range(M):
			for ix in range(M):
				index = ix + iy*M + iz*M2
				grain_id = grain_ids[index]

				# right, left, forward, backward, up, down
				nb_indices[0] = int_mod(ix+1, M) + iy*M + iz*M2
				nb_indices[1] = int_mod(ix-1, M) + iy*M + iz*M2
				nb_indices[2] = ix + int_mod(iy+1, M)*M + iz*M2
				nb_indices[3] = ix + int_mod(iy-1, M)*M + iz*M2
				nb_indices[4] = ix + iy*M + int_mod(iz+1, M)*M2
				nb_indices[5] = ix + iy*M + int_mod(iz-1, M)*M2
				
				for i in range(6):
					nb_id = grain_ids[nb_indices[i]]
					if nb_id != grain_id:
						surface_voxels[index] += 1
						if grain_id > 1: # only consider WC here
							if nb_id == 1:
								interface_voxels[index] += 1
							else:
								gb_voxels[index] += 1
	
	return surface_voxels, gb_voxels, interface_voxels
