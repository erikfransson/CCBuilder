#!/usr/bin/python2
import numpy as np
import CCBuilder as ccb
import CCBuilder_c as ccb_c
import cPickle as pickle
import time

print "Running"
vol_frac_goal = 0.8
# Cube size:
L = 5.

# Number of voxels
M = 120
delta_x = L/M

mc_steps = 100*M**3
kBT = 0.5

nr_tries = 2500

# Move max 0.75 mu in any direction
delta = int(0.75 / delta_x)
nr_tries = 2500

# to avoid confusion
vol_frac_goal = np.double(vol_frac_goal)
L = np.double(L)
M = np.int(M)
mc_steps = np.int(mc_steps)
kBT = np.double(kBT)
nr_tries = np.int(nr_tries)

trunc_triangles = ccb.prepare_triangles(vol_frac_goal, L)
ccb.optimize_midpoints(L, trunc_triangles)

with open('trunc_triangles_0.data', 'wb') as f:
	pickle.dump(L, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(trunc_triangles, f, pickle.HIGHEST_PROTOCOL)

voxel_indices_xyz = ccb_c.make_voxel_indices(L, M, trunc_triangles)
grain_ids_0, overlaps_0, voxel_indices_0 = ccb_c.populate_voxels(M, voxel_indices_xyz, nr_tries, delta)

phases_0, good_voxels_0, euler_angles_0, phase_volumes_0, grain_volumes_0 = ccb_c.calc_grain_prop(M, grain_ids_0, trunc_triangles)
surface_voxels_0, gb_voxels_0, interface_voxels_0 = ccb_c.calc_surface_prop(M, grain_ids_0)

vol_frac_WC_0 = phase_volumes_0[1]/np.float(np.sum(phase_volumes_0))
vol_frac_Co_0 = 1 - vol_frac_WC_0
mass_frac_WC_0 = ccb.mass_fraction(vol_frac_WC_0)
d_eq_0 = ccb.volume_to_eq_d(grain_volumes_0*delta_x**3)

sum_gb_voxels_0 = np.sum(gb_voxels_0)
contiguity_0 = sum_gb_voxels_0 / np.float(sum_gb_voxels_0 + np.sum(interface_voxels_0))

ccb.write_hdf5('testfile_0.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_0, phases_0, good_voxels_0, euler_angles_0, surface_voxels_0, gb_voxels_0, interface_voxels_0, overlaps_0)

# Make new copies to play with
grain_ids_1 = grain_ids_0.copy()
gb_voxels_1 = gb_voxels_0.copy()

start_time = time.time()
ccb_c.make_mcp_unlim(M, grain_ids_1, gb_voxels_1, mc_steps, kBT)
print np.str(time.time() - start_time) + " seconds"

surface_voxels_1, gb_voxels_1, interface_voxels_1 = ccb_c.calc_surface_prop(M, grain_ids_1)
phases_1, good_voxels_1, euler_angles_1, phase_volumes_1, grain_volumes_1 = ccb_c.calc_grain_prop(M, grain_ids_1, trunc_triangles)

vol_frac_WC_1 = phase_volumes_1[1]/np.float(np.sum(phase_volumes_1))
vol_frac_Co_1 = 1 - vol_frac_WC_1
mass_frac_WC_1 = ccb.mass_fraction(vol_frac_WC_1)
d_eq_1 = ccb.volume_to_eq_d(grain_volumes_1*delta_x**3)

sum_gb_voxels_1 = np.sum(gb_voxels_1)
contiguity_1 = sum_gb_voxels_1 / np.float(sum_gb_voxels_1 + np.sum(interface_voxels_1))

ccb.write_hdf5('testfile_1.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_1, phases_1, good_voxels_1, euler_angles_1, surface_voxels_1, gb_voxels_1, interface_voxels_1, overlaps_0)

# Make new copies to play with
grain_ids_2 = grain_ids_0.copy()
gb_voxels_2 = gb_voxels_0.copy()

start_time = time.time()
ccb_c.make_mcp_bound(M, grain_ids_2, gb_voxels_2, voxel_indices_0, mc_steps, kBT)
print np.str(time.time() - start_time) + " seconds"

surface_voxels_2, gb_voxels_2_1, interface_voxels_2 = ccb_c.calc_surface_prop(M, grain_ids_2)
phases_2, good_voxels_2, euler_angles_2, phase_volumes_2, grain_volumes_2 = ccb_c.calc_grain_prop(M, grain_ids_2, trunc_triangles)

vol_frac_WC_2 = phase_volumes_2[1]/np.float(np.sum(phase_volumes_2))
vol_frac_Co_2 = 1 - vol_frac_WC_2
mass_frac_WC_2 = ccb.mass_fraction(vol_frac_WC_2)
d_eq_2 = ccb.volume_to_eq_d(grain_volumes_2*delta_x**3)

sum_gb_voxels_2 = np.sum(gb_voxels_2)
contiguity_2 = sum_gb_voxels_2 / np.float(sum_gb_voxels_2 + np.sum(interface_voxels_2))

ccb.write_hdf5('testfile_2.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_2, phases_2, good_voxels_2, euler_angles_2, surface_voxels_2, gb_voxels_2, interface_voxels_2, overlaps_0)


ccb.write_oofem('testfile_2', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_2)
