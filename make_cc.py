import numpy as np
import CCBuilder as ccb
import CCBuilder_c as ccb_c
import cPickle as pickle
import time

vol_frac_goal = 0.80; L = 4.2

trunc_triangles = ccb.prepare_triangles(vol_frac_goal, L)

start_time = time.time()
trunc_triangles_list, neighbors = ccb.make_neighbors(trunc_triangles, L)
print np.str(time.time() - start_time) + " seconds"

x = np.array([trunc_triangle[0].midpoint for trunc_triangle in trunc_triangles_list])
grain_volumes_0 = [trunc_triangle[0].volume for trunc_triangle in trunc_triangles_list]
d_eq_0 = [trunc_triangle[0].d_eq for trunc_triangle in trunc_triangles_list]
circumcircle = [trunc_triangle[0].circumcircle for trunc_triangle in trunc_triangles_list]

with open('trunc_triangles.data', 'wb') as f:
	pickle.dump(L, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(trunc_triangles_list, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(neighbors, f, pickle.HIGHEST_PROTOCOL)

M = 100
delta_x = L/M

start_time = time.time()
grain_ids, phases, good_voxels, euler_angles, phase_volumes, grain_volumes = ccb_c.populate_voxels(L, M, trunc_triangles_list)
print np.str(time.time() - start_time) + " seconds"

start_time = time.time()
surface_voxels, gb_voxels, interface_voxels = ccb_c.calc_surface_prop(M, grain_ids)
print np.str(time.time() - start_time) + " seconds"

vol_frac_WC = phase_volumes[1]/np.float(np.sum(phase_volumes))
vol_frac_Co = 1 - vol_frac_WC
mass_frac_WC = ccb.mass_fraction(vol_frac_WC)
d_eq = ccb.volume_to_eq_d(grain_volumes*delta_x**3)

sum_gb_voxels = np.sum(gb_voxels)
contiguity = sum_gb_voxels / np.float(sum_gb_voxels + np.sum(interface_voxels))

ccb.write_hdf5('testfile_with_sv.hdf5', 3*[M], 3*[delta_x], trunc_triangles_list, grain_ids, phases, good_voxels, euler_angles, surface_voxels)
ccb.write_hdf5('testfile_wo_sv.hdf5', 3*[M], 3*[delta_x], trunc_triangles_list, grain_ids, phases, good_voxels, euler_angles)
