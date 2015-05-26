# cython: boundscheck=False, wraparound=False, profile=False, cdivision=True

import cython
import numpy as np
cimport numpy as np
import random
import TruncatedTriangle

from libc.math cimport sqrt, floor, exp
from libc.stdlib cimport rand, RAND_MAX, srand, malloc, free
from libc.limits cimport INT_MAX

cdef inline int int_max(int a, int b):
	return a if a >= b else b

cdef inline int int_min(int a, int b):
	return a if a <= b else b

cdef inline double double_max(double a, double b):
	return a if a >= b else b

cdef inline double mod(double x1, double x2):
	return x1 - floor(x1 / x2) * x2

cdef inline double random_double():
	return <double> rand() / <double> RAND_MAX

# Assume that i2 > 0
# C %-operator does not behave like I want it to
cdef inline int int_mod(int i1, int i2):
	return i1%i2 if i1 >= 0 else ((i1%i2) + i2) % i2

cdef inline long long longlong_mod(long long i1, long long i2):
	return i1%i2 if i1 >= 0 else ((i1%i2) + i2) % i2

# Returns a random integer in [min_i, max_i-1]
cdef unsigned int rand_interval(unsigned int min_i, unsigned int max_i):
	cdef unsigned int r
	cdef unsigned int interval = max_i - min_i
	cdef unsigned int buckets = RAND_MAX / interval
	cdef unsigned int limit = buckets * interval
	
	while True:
		r = rand();
		if r < limit:
			break
	
	return min_i + (r / buckets)

# The elements in A must be ordered
cdef bint binary_search(int* A, int key, int imin, int imax):
	cdef int imid
	
	# continue searching while [imin,imax] is not empty
	while (imax >= imin):
		# calculate the midpoint for roughly equal partition
		imid = imin + (imax - imin) / 2
		if A[imid] == key:
			# key found at index imid
			return True; 
		# determine which subarray to search
		elif A[imid] < key:
			# change min index to search upper subarray
			imin = imid + 1
		else:
		# change max index to search lower subarray
			imax = imid - 1
	# key was not found
	return False;

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
							grad = volume[i]*volume[j]*2.0/(r_0*r_0) * (1.0 - r_0/r_ij_norm)
							U_grad[3*i] -= grad*r_ijx
							U_grad[3*i+1] -= grad*r_ijy
							U_grad[3*i+2] -= grad*r_ijz
							U_grad[3*j] += grad*r_ijx
							U_grad[3*j+1] += grad*r_ijy
							U_grad[3*j+2] += grad*r_ijz
	
	return U, U_grad

def make_voxel_indices(double L, int M, list trunc_triangles):
	print "Making a list of voxel indices (ix,iy,iz) for each grain."
	
	cdef:
		int i, min_ix, max_ix, min_iy, max_iy, min_iz, max_iz, ix, iy, iz
		int M2 = M*M
		int M3 = M2*M
		double delta_x = L/M, t
		
		# Needed to calculate if inside
		np.ndarray[double, ndim=2, mode="c"] vert
		np.ndarray[double, ndim=1, mode="c"] midpoint
		np.ndarray[double, ndim=2, mode="c"] rot_matrix_tr

		double r0[3]
		double r1[3]
		
		bint truncation, inside
		
		list voxel_indices = [], voxel_indices_i
	
	for i,tr_tri in enumerate(trunc_triangles):
		vert = tr_tri.vertices
		
		midpoint = tr_tri.midpoint
		rot_matrix_tr = tr_tri.rot_matrix_tr
		t = tr_tri.t
		truncation = tr_tri.r > 0
		
		min_ix = <int> floor(M*tr_tri.min_x/L)
		max_ix = <int> floor(M*tr_tri.max_x/L)
		min_iy = <int> floor(M*tr_tri.min_y/L)
		max_iy = <int> floor(M*tr_tri.max_y/L)
		min_iz = <int> floor(M*tr_tri.min_z/L)
		max_iz = <int> floor(M*tr_tri.max_z/L)
		
		voxel_indices_i = []
		
		for iz in range(min_iz, max_iz+1):
			for iy in range(min_iy, max_iy+1):
				for ix in range(min_ix, max_ix+1):
					r0[0] = delta_x*(0.5+ix) - midpoint[0]
					r0[1] = delta_x*(0.5+iy) - midpoint[1]
					r0[2] = delta_x*(0.5+iz) - midpoint[2]
					
					# r1 is r0 expressed in coordinates fixed in the triangle
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
						voxel_indices_i.append(int_mod(ix, M))
						voxel_indices_i.append(int_mod(iy, M))
						voxel_indices_i.append(int_mod(iz, M))
		
		voxel_indices.append(np.array(voxel_indices_i, dtype="int32"))
	
	return voxel_indices

# Make N_tries attempts to place each grain with minimum overlap with existing grains. The position is varied randomly within [-delta,+delta] from the original position. The function will not touch the list voxel_indices_xyz and it will hence not be altered by the attempts. Returns a sorted list of indices (which is not separated in x,y,z components) to use in make_mcp_bound. The indices of the returned list correspond to optimal grain positions.
def populate_voxels(int M, double L, list trunc_triangles, int N_tries, int delta):
	print "Populating voxels"

	# Seed rand with something
	srand(random.randint(0, INT_MAX))

	voxel_indices_xyz = make_voxel_indices(L, M, trunc_triangles)

	cdef:
		int i, j, M2 = M*M, M3 = M2*M, ix, iy, iz, index, N_voxels
		int N_grains = len(voxel_indices_xyz)
		np.ndarray[int, ndim=1, mode="c"] grain_ids = np.ones(dtype="int32", shape=(M3))
		np.ndarray[char, ndim=1, mode="c"] overlaps = np.zeros(dtype="int8", shape=(M3))
		int **voxel_indices_c
		int *voxel_indices_c_len
		np.ndarray[int, ndim=1, mode="c"] voxel_indices_xyz_i
		int overlap_j, overlap_min, n_tries, delta_x, delta_y, delta_z, delta_x_j, delta_y_j, delta_z_j
		list voxel_indices = []
		np.ndarray[int, ndim=1, mode="c"] voxel_indices_i

	voxel_indices_c = <int**> malloc(N_grains*sizeof(int*))
	if not voxel_indices_c:
		raise MemoryError()
	voxel_indices_c_len = <int*> malloc(N_grains*sizeof(int))
	if not voxel_indices_c_len:
		raise MemoryError()
	
	for i in range(N_grains):
		voxel_indices_xyz_i = voxel_indices_xyz[i]
		voxel_indices_c_len[i] = <int> voxel_indices_xyz_i.shape[0] / 3
		# an int pointer to the first element in voxel_indices_xyz_i
		voxel_indices_c[i] = &voxel_indices_xyz_i[0]
	
	for i in range(N_grains):
		N_voxels = voxel_indices_c_len[i]
		n_tries = 0
		overlap_min = INT_MAX
		
		delta_x = 0
		delta_y = 0
		delta_z = 0
		
		if N_tries > 0 and delta > 0:
			# do,while loops would be nice
			while True:
				if n_tries == 0:
					delta_x_j = 0
					delta_y_j = 0
					delta_z_j = 0
				else:
					delta_x_j = -delta + rand_interval(0, 2*delta+1)
					delta_y_j = -delta + rand_interval(0, 2*delta+1)
					delta_z_j = -delta + rand_interval(0, 2*delta+1)
				
				overlap_j = 0
				for j in range(N_voxels):
					ix = int_mod(voxel_indices_c[i][3*j] + delta_x_j, M)
					iy = int_mod(voxel_indices_c[i][3*j+1] + delta_y_j, M)
					iz = int_mod(voxel_indices_c[i][3*j+2] + delta_z_j, M)
					index = ix + iy*M + iz*M2
					if grain_ids[index] > 1: # claimed, so add overlap
						overlap_j += 1
				
				if overlap_j < overlap_min:
					overlap_min = overlap_j
					delta_x = delta_x_j
					delta_y = delta_y_j
					delta_z = delta_z_j
				
				n_tries += 1
				if overlap_min == 0 or n_tries == N_tries:
					break
		
		print "grain {}: tries: {} delta: {} {} {}".format(*(i, n_tries, delta_x, delta_y, delta_z))
		
		voxel_indices_i = np.zeros(dtype="int32", shape=(N_voxels))
		
		# Rerun with optimal delta_x, delta_y, delta_z
		for j in range(N_voxels):
			ix = int_mod(voxel_indices_c[i][3*j] + delta_x, M)
			iy = int_mod(voxel_indices_c[i][3*j+1] + delta_y, M)
			iz = int_mod(voxel_indices_c[i][3*j+2] + delta_z, M)
			# Do NOT update voxel_indices_xyz
			#voxel_indices_c[i][3*j] = ix
			#voxel_indices_c[i][3*j+1] = iy
			#voxel_indices_c[i][3*j+2] = iz
			index = ix + iy*M + iz*M2
			voxel_indices_i[j] = index
			if grain_ids[index] == 1: # still unclaimed binder
				grain_ids[index] = i+2
			elif grain_ids[index] > 1: # claimed, so add overlap
				overlaps[index] += 1
		
		voxel_indices_i.sort()
		voxel_indices.append(voxel_indices_i)

		# Move the truncated triangle to the right position as well:
		trunc_triangles[i].midpoint += [delta_x, delta_y, delta_z]
	
	free(voxel_indices_c)
	free(voxel_indices_c_len)
	
	return grain_ids, overlaps, voxel_indices

# Returns grain properties.
# Phases is a list of all voxels, 1 if binder, 2 if grain.
# Good_voxels does nothing?
# euler_angles contains the three euler angles for each voxel, 0 if the voxel is binder
# phases_volumes contains the Co and the WC volume ( in voxel counts )
# grain_volumes contains the volume for each grain ( in voxel counts )
def calc_grain_prop(int M, np.ndarray[int, ndim=1] grain_ids, list trunc_triangles):
	print "Populating grain and voxel properties"
	
	cdef int M3, grain_id, N, index
	M3 = M*M*M
	
	cdef np.ndarray[int, ndim=1] phases = np.ones(dtype="int32", shape=(M3))
	cdef np.ndarray[unsigned char, ndim=1] good_voxels = np.ones(dtype="uint8", shape=(M3))
	cdef np.ndarray[float, ndim=2] euler_angles = np.zeros(dtype="float32", shape=(M3, 3))
	
	N = len(trunc_triangles)
	cdef np.ndarray[int, ndim=1] grain_volumes = np.zeros(dtype="int32", shape=(N))
	
	cdef np.ndarray[float, ndim=2, mode="c"] euler_angles_j = np.zeros(dtype="float32", shape=(N, 3))
	
	for index in range(N):
		euler_angles_j[index, 0] = <float> trunc_triangles[index].euler_angles[0]
		euler_angles_j[index, 1] = <float> trunc_triangles[index].euler_angles[1]
		euler_angles_j[index, 2] = <float> trunc_triangles[index].euler_angles[2]
	
	cdef np.ndarray[int, ndim=1] phase_volumes = np.zeros(dtype="int32", shape=(2))
	phase_volumes[0] = M3
	phase_volumes[1] = 0
	
	for index in range(M3):
		if grain_ids[index] > 1:
			grain_id = grain_ids[index] - 2
			phases[index] = 2 # WC
			euler_angles[index, 0] = euler_angles_j[grain_id, 0]
			euler_angles[index, 1] = euler_angles_j[grain_id, 1]
			euler_angles[index, 2] = euler_angles_j[grain_id, 2]
			phase_volumes[0] -= 1
			phase_volumes[1] += 1
			grain_volumes[grain_id] += 1
	
	return phases, good_voxels, euler_angles, phase_volumes, grain_volumes

def calc_gb_indices(np.ndarray[char, ndim=1] gb_voxels):
	print "Populating grain boundary indices"
	
	cdef int i
	cdef list gb_indices = []
	for i in range(0, gb_voxels.shape[0]):
		if gb_voxels[i]:
			gb_indices.append(i)
	return gb_indices

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
							if nb_id == 1: # binder
								interface_voxels[index] += 1
							else: # nb_id > 1, WC
								gb_voxels[index] += 1
	
	return surface_voxels, gb_voxels, interface_voxels

def calc_mli(int M, double L, np.ndarray[int, ndim=1] grain_ids):
	print "Calculating mean linear intercept"
	
	cdef int ix, iy, iz, old_grain_id, grain_id, M2, M3
	cdef int N_gb_x = 0, N_interf_x = 0, N_gb_y = 0, N_interf_y = 0, N_gb_z = 0, N_interf_z = 0, N_gb, N_interf
	cdef double cont, vol_frac_WC, vol_frac_Co, d_WC, d_Co
	M2 = M*M
	M3 = M2*M
	cdef int* phase_volumes = [M3, 0]
	
	# along x-axis
	for iz in range(M):
		for iy in range(M):
			old_grain_id = grain_ids[M-1 + iy*M + iz*M2]
			for ix in range(0, M):
				index = ix + iy*M + iz*M2
				grain_id = grain_ids[index]
				if grain_id != old_grain_id:
					if grain_id == 1 or old_grain_id == 1: # to or from binder, i.e. WC/binder interface
						N_interf_x += 1
					else: # grain_id > 1, WC
						N_gb_x += 1
				if grain_id > 1:
					phase_volumes[1] += 1
					phase_volumes[0] -= 1
				
				old_grain_id = grain_id
	
	# along y-axis
	for iz in range(M):
		for ix in range(M):
			old_grain_id = grain_ids[ix + (M-1)*M + iz*M2]
			for iy in range(0, M):
				index = ix + iy*M + iz*M2
				grain_id = grain_ids[index]
				if grain_id != old_grain_id:
					if grain_id == 1 or old_grain_id == 1: # to or from binder, i.e. WC/binder interface
						N_interf_y += 1
					else: # grain_id > 1, WC
						N_gb_y += 1
				old_grain_id = grain_id
	
	# along z-axis
	for iy in range(M):
		for ix in range(M):
			old_grain_id = grain_ids[ix + iy*M + (M-1)*M2]
			for iz in range(0, M):
				index = ix + iy*M + iz*M2
				grain_id = grain_ids[index]
				if grain_id != old_grain_id:
					if grain_id == 1 or old_grain_id == 1: # to or from binder, i.e. WC/binder interface
						N_interf_z += 1
					else: # grain_id > 1, WC
						N_gb_z += 1
				old_grain_id = grain_id
	
	N_gb = N_gb_x + N_gb_y + N_gb_z
	N_interf = N_interf_x + N_interf_y + N_interf_z
	
	cont = 2*N_gb / <double> (2*N_gb + N_interf)
	
	vol_fraction_WC = phase_volumes[1] / <double> M3
	vol_fraction_Co = phase_volumes[0] / <double> M3
	
	# mean free path of WC and Co, resp.
	d_WC = 2*vol_fraction_WC / (2*N_gb + N_interf) * L * (3*M2)
	d_Co = 2*vol_fraction_Co / N_interf * L * (3*M2)
	
	return cont, d_WC, d_Co

# Cleanup of stray voxels that appear due to discretization errors.
def stray_cleanup(int M, np.ndarray[int, ndim=1] grain_ids, int min_n=3, int iterations=3):
	print "Stray voxel cleanup"
	
	cdef:
		int M2, M3, i, gb_voxel_index, gb_voxel_id, nb_id, ix, iy, iz
		int nb_max
		int n
	
	nb_indices = np.zeros(18, dtype=int)
	nb_ids = np.zeros(18, dtype=int)
	nb_count = np.zeros(18, dtype=int)
	M2 = M*M
	M3 = M2*M
	
	for iteration in range(iterations):
		print("iter", iteration)
		for ix in range(M):
			for iy in range(M):
				for iz in range(M):
					gb_voxel_index = ix + iy*M + iz*M2
					gb_voxel_id = grain_ids[gb_voxel_index]
				
					# right, left, forward, backward, up, down - neighbours
					nb_indices[0] = int_mod(ix+1, M) + iy*M + iz*M2
					nb_indices[1] = int_mod(ix-1, M) + iy*M + iz*M2
					nb_indices[2] = ix + int_mod(iy+1, M)*M + iz*M2
					nb_indices[3] = ix + int_mod(iy-1, M)*M + iz*M2
					# corners
					nb_indices[4] = int_mod(ix+1, M) + int_mod(iy+1, M)*M + iz*M2
					nb_indices[5] = int_mod(ix+1, M) + int_mod(iy-1, M)*M + iz*M2
					nb_indices[6] = int_mod(ix-1, M) + int_mod(iy+1, M)*M + iz*M2
					nb_indices[7] = int_mod(ix-1, M) + int_mod(iy-1, M)*M + iz*M2
					# Layer above
					nb_indices[8] = ix + iy*M + int_mod(iz+1, M)*M2
					nb_indices[9] = ix + int_mod(iy+1, M)*M + int_mod(iz+1, M)*M2
					nb_indices[10] = ix + int_mod(iy-1, M)*M + int_mod(iz+1, M)*M2
					nb_indices[11] = int_mod(ix+1, M) + iy*M + int_mod(iz+1, M)*M2
					nb_indices[12] = int_mod(ix-1, M) + iy*M + int_mod(iz+1, M)*M2
					# Layer below
					nb_indices[13] = ix + iy*M + int_mod(iz-1, M)*M2
					nb_indices[14] = ix + int_mod(iy+1, M)*M + int_mod(iz-1, M)*M2
					nb_indices[15] = ix + int_mod(iy-1, M)*M + int_mod(iz-1, M)*M2
					nb_indices[16] = int_mod(ix+1, M) + iy*M + int_mod(iz-1, M)*M2
					nb_indices[17] = int_mod(ix-1, M) + iy*M + int_mod(iz-1, M)*M2
				
					# Clear and populate the *set* of different neighbor indices that are allowed; id > 1, i.e. not binder
					n = 0
					for i in range(18):
						nb_ids[i] = grain_ids[nb_indices[i]]
						if gb_voxel_id == nb_ids[i]:
							n += 1
	
					if n <= min_n: # Checking for 1 or 0 neighbours.
						# Find the most common neighbor and use that instead.
						for i in range(18):
							nb_count[i] = 0
							for j in range(18):
								if nb_ids[i] == nb_ids[j]:
									nb_count[i] += 1
						surrounding_id = nb_ids[nb_count.argmax()]
						print("switch", grain_ids[gb_voxel_index], "to", surrounding_id)
						grain_ids[gb_voxel_index] = surrounding_id
	
# Monte Carlo of the Potts model with unlimited grains. gb_voxels must be consistent with grain_ids.
def make_mcp_unlim(int M, np.ndarray[int, ndim=1] grain_ids, np.ndarray[char, ndim=1] gb_voxels, int steps, double kBT):
	print "Making Monte Carlo steps"
	
	cdef:
		int M2, M3, i, step, gb_voxel_index, gb_voxel_id, nb_id, ix, iy, iz, new_id, sum_delta_A, nr_diff_ids
		int nb_indices[6]
		int nb_ids[6]
		int nb_set[6]
		int delta_A[6]
		double exp_dA_kBT[4]
		double random_number
	
	M2 = M*M
	M3 = M2*M
	
	# Seed rand with something
	srand(random.randint(0, INT_MAX))
	
	# Set the exponentials
	for i in range(4):
		exp_dA_kBT[i] = exp(-(i+1)/kBT)
	
	for step in range(steps):
		# Choose a random voxel and check if it is a gb voxel
		gb_voxel_index = rand_interval(0, M3)
		
		if gb_voxels[gb_voxel_index]:
			gb_voxel_id = grain_ids[gb_voxel_index]
			
			iz = gb_voxel_index / M2
			iy = (gb_voxel_index - iz*M2) / M
			ix = gb_voxel_index - iz*M2 - iy*M
			
			# right, left, forward, backward, up, down
			nb_indices[0] = int_mod(ix+1, M) + iy*M + iz*M2
			nb_indices[1] = int_mod(ix-1, M) + iy*M + iz*M2
			nb_indices[2] = ix + int_mod(iy+1, M)*M + iz*M2
			nb_indices[3] = ix + int_mod(iy-1, M)*M + iz*M2
			nb_indices[4] = ix + iy*M + int_mod(iz+1, M)*M2
			nb_indices[5] = ix + iy*M + int_mod(iz-1, M)*M2
			
			# Clear and populate the *set* of different neighbor indices that are allowed; id > 1, i.e. not binder
			for i in range(6):
				nb_ids[i] = grain_ids[nb_indices[i]]
				nb_set[i] = -1
			nr_diff_ids = 0
			for i in range(6):
				nb_id = nb_ids[i]
				if nb_id > 1 and nb_id != gb_voxel_id and nb_id != nb_set[0] and nb_id != nb_set[1] and nb_id != nb_set[2] and nb_id != nb_set[3] and nb_id != nb_set[4] and nb_id != nb_set[5]:
					nb_set[nr_diff_ids] = nb_id
					nr_diff_ids += 1
			
			if nr_diff_ids > 1:
				new_id = nb_set[rand_interval(0, nr_diff_ids)]
			else:
				new_id = nb_set[0]
			
			sum_delta_A = 0
			for i in range(6):
				if nb_ids[i] == gb_voxel_id: # the new id is different, add 1 area
					delta_A[i] = 1
					sum_delta_A += 1
				elif nb_ids[i] == new_id: # the new id is the same, subtract 1 area
					delta_A[i] = -1
					sum_delta_A -= 1
				else:
					delta_A[i] = 0
			
			if sum_delta_A < 0 or (sum_delta_A == 0 and random_double() < 0.5) or (sum_delta_A > 0 and random_double() < exp_dA_kBT[sum_delta_A-1]):
				grain_ids[gb_voxel_index] = new_id
				gb_voxels[gb_voxel_index] += sum_delta_A
				for i in range(6):
					gb_voxels[nb_indices[i]] += delta_A[i]

# Monte Carlo of the Potts model where the grains are bound to regions where two or more grains overlap. Quite useless since overlaps tend to be everywhere at high WC fraction. gb_voxels must be consistent with grain_ids.
def make_mcp_overlap(int M, np.ndarray[int, ndim=1] grain_ids, np.ndarray[char, ndim=1] gb_voxels, np.ndarray[char, ndim=1] overlaps, int steps, double kBT):
	print "Making Monte Carlo steps"
	
	cdef:
		int M2, M3, i, step, gb_voxel_index, gb_voxel_id, nb_id, ix, iy, iz, new_id, sum_delta_A, nr_diff_ids
		int nb_indices[6]
		int nb_ids[6]
		int nb_set[6]
		int delta_A[6]
		double exp_dA_kBT[4]
		double random_number
	
	M2 = M*M
	M3 = M2*M
	
	# Seed rand with something
	srand(random.randint(0, INT_MAX))
	
	# Set the exponentials
	for i in range(4):
		exp_dA_kBT[i] = exp(-(i+1)/kBT)
	
	for step in range(steps):
		# Choose a random voxel and check if it is a gb voxel and an overlap voxel
		gb_voxel_index = rand_interval(0, M3)
		
		if gb_voxels[gb_voxel_index] and overlaps[gb_voxel_index]:
			gb_voxel_id = grain_ids[gb_voxel_index]
			
			iz = gb_voxel_index / M2
			iy = (gb_voxel_index - iz*M2) / M
			ix = gb_voxel_index - iz*M2 - iy*M
			
			# right, left, forward, backward, up, down
			nb_indices[0] = int_mod(ix+1, M) + iy*M + iz*M2
			nb_indices[1] = int_mod(ix-1, M) + iy*M + iz*M2
			nb_indices[2] = ix + int_mod(iy+1, M)*M + iz*M2
			nb_indices[3] = ix + int_mod(iy-1, M)*M + iz*M2
			nb_indices[4] = ix + iy*M + int_mod(iz+1, M)*M2
			nb_indices[5] = ix + iy*M + int_mod(iz-1, M)*M2
			
			# Clear and populate the *set* of different neighbor indices that are allowed; id > 1, i.e. not binder
			for i in range(6):
				nb_ids[i] = grain_ids[nb_indices[i]]
				nb_set[i] = -1
			nr_diff_ids = 0
			for i in range(6):
				nb_id = nb_ids[i]
				if nb_id > 1 and nb_id != gb_voxel_id and nb_id != nb_set[0] and nb_id != nb_set[1] and nb_id != nb_set[2] and nb_id != nb_set[3] and nb_id != nb_set[4] and nb_id != nb_set[5]:
					nb_set[nr_diff_ids] = nb_id
					nr_diff_ids += 1
			
			if nr_diff_ids > 1:
				new_id = nb_set[rand_interval(0, nr_diff_ids)]
			else:
				new_id = nb_set[0]
			
			sum_delta_A = 0
			for i in range(6):
				if nb_ids[i] == gb_voxel_id: # the new id is different, add 1 area
					delta_A[i] = 1
					sum_delta_A += 1
				elif nb_ids[i] == new_id: # the new id is the same, subtract 1 area
					delta_A[i] = -1
					sum_delta_A -= 1
				else:
					delta_A[i] = 0
			
			if sum_delta_A < 0 or (sum_delta_A == 0 and random_double() < 0.5) or (sum_delta_A > 0 and random_double() < exp_dA_kBT[sum_delta_A-1]):
				grain_ids[gb_voxel_index] = new_id
				gb_voxels[gb_voxel_index] += sum_delta_A
				for i in range(6):
					gb_voxels[nb_indices[i]] += delta_A[i]

# Monte Carlo of the Potts model where the grains are bound to their original truncated triangle shape. gb_voxels must be consistent with grain_ids.
def make_mcp_bound(int M, np.ndarray[int, ndim=1] grain_ids, np.ndarray[char, ndim=1] gb_voxels, list voxel_indices, long long steps, double kBT):
	print "Making Monte Carlo steps"
	
	cdef:
		int M2, M3, i, j, gb_voxel_index, gb_voxel_id, nb_id, ix, iy, iz, new_id, sum_delta_A, nr_diff_ids, N
		bint in_set
		int nb_indices[6]
		int nb_ids[6]
		int nb_set[6]
		int delta_A[6]
		double exp_dA_kBT[4]
		double random_number
		int **voxel_indices_c
		int *voxel_indices_c_len
		np.ndarray[int, ndim=1, mode="c"] voxel_indices_c_i
		long long step, steps_10, steps_100
	
	steps_10 = steps / 10
	steps_100 = steps / 100
	
	N = len(voxel_indices)
	voxel_indices_c = <int**> malloc(N*sizeof(int*))
	if not voxel_indices_c:
		raise MemoryError()
	voxel_indices_c_len = <int*> malloc(N*sizeof(int))
	if not voxel_indices_c_len:
		raise MemoryError()
	
	for i in range(N):
		voxel_indices_c_i = voxel_indices[i]
		voxel_indices_c_len[i] = <int> voxel_indices_c_i.shape[0]
		# an int pointer to the first element in voxel_indices_c_i
		voxel_indices_c[i] = &voxel_indices_c_i[0]
	
	M2 = M*M
	M3 = M2*M
	
	# Seed rand with something
	srand(random.randint(0, INT_MAX))
	
	# Set the exponentials
	for i in range(4):
		exp_dA_kBT[i] = exp(-(i+1)/kBT)
	
	for step in range(steps):
		if longlong_mod(step, steps_10) == 0:
			print np.str(step / steps_100) + "%, step " + np.str(step)
		
		# Choose a random voxel
		gb_voxel_index = rand_interval(0, M3)
		
		# Check if it is a gb voxel
		if gb_voxels[gb_voxel_index]:
			gb_voxel_id = grain_ids[gb_voxel_index]
			
			iz = gb_voxel_index / M2
			iy = (gb_voxel_index - iz*M2) / M
			ix = gb_voxel_index - iz*M2 - iy*M
			
			# right, left, forward, backward, up, down
			nb_indices[0] = int_mod(ix+1, M) + iy*M + iz*M2
			nb_indices[1] = int_mod(ix-1, M) + iy*M + iz*M2
			nb_indices[2] = ix + int_mod(iy+1, M)*M + iz*M2
			nb_indices[3] = ix + int_mod(iy-1, M)*M + iz*M2
			nb_indices[4] = ix + iy*M + int_mod(iz+1, M)*M2
			nb_indices[5] = ix + iy*M + int_mod(iz-1, M)*M2
			
			# Clear and populate the *set* of different neighbor indices that are allowed; id > 1, i.e. not binder
			for i in range(6):
				nb_ids[i] = grain_ids[nb_indices[i]]
				nb_set[i] = -1
			nr_diff_ids = 0
			for i in range(6):
				nb_id = nb_ids[i]
				if nb_id > 1 and nb_id != gb_voxel_id: # Not binder and not the same id
					in_set = False
					for j in range(nr_diff_ids):
						in_set = nb_id == nb_set[j]
						if in_set:
							break
					# Not in set and gb_voxel_id belongs to nb_id and can thus be changed to nb_id
					if not in_set and binary_search(voxel_indices_c[nb_id-2], gb_voxel_index, 0, voxel_indices_c_len[nb_id-2]-1):
						nb_set[nr_diff_ids] = nb_id
						nr_diff_ids += 1
			
			# the set of allowed changes can be zero if we are at the edges of a truncated triangle
			if nr_diff_ids > 0:
				if nr_diff_ids > 1:
					new_id = nb_set[rand_interval(0, nr_diff_ids)]
				else:
					new_id = nb_set[0]
				
				sum_delta_A = 0
				for i in range(6):
					if nb_ids[i] == gb_voxel_id: # the new id is different, add 1 area
						delta_A[i] = 1
						sum_delta_A += 1
					elif nb_ids[i] == new_id: # the new id is the same, subtract 1 area
						delta_A[i] = -1
						sum_delta_A -= 1
					else:
						delta_A[i] = 0
				
				# Metropolis algorithm
				if sum_delta_A < 0 or (sum_delta_A == 0 and random_double() < 0.5) or (sum_delta_A > 0 and random_double() < exp_dA_kBT[sum_delta_A-1]):
					grain_ids[gb_voxel_index] = new_id
					gb_voxels[gb_voxel_index] += sum_delta_A
					for i in range(6):
						gb_voxels[nb_indices[i]] += delta_A[i]
	
	free(voxel_indices_c)
	free(voxel_indices_c_len)
