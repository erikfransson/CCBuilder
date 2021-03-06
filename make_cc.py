#!/usr/bin/python2
import numpy as np
import CCBuilder as ccb
import CCBuilder_c as ccb_c
import cPickle as pickle
import time
import matplotlib.pyplot as plt
from misorientation import *

print "Running"
vol_frac_goal = 1.0
# Cube size:
L = 5.
np.random.seed(0)

M = 50
delta_x = L/M

mc_steps = 100*M**3
kBT = 0.5


# Move max 0.75 mu in any direction
delta = M #int(0.75 / delta_x)
nr_tries = 1000

# to avoid confusion
vol_frac_goal = np.double(vol_frac_goal)
L = np.double(L)
M = np.int(M)
mc_steps = np.int(mc_steps)
kBT = np.double(kBT)
nr_tries = np.int(nr_tries)

trunc_triangles = ccb.prepare_triangles(1.0, L) # Using 1.0 instead of vol_frac_goal in order to obtain enough inputs.
#ccb.optimize_midpoints(L, trunc_triangles)

with open('trunc_triangles_0.data', 'wb') as f:
	pickle.dump(L, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(trunc_triangles, f, pickle.HIGHEST_PROTOCOL)

grain_ids_0, overlaps_0, voxel_indices_0 = ccb_c.populate_voxels(M, L, trunc_triangles, nr_tries, delta, vol_frac_goal)

phases_0, good_voxels_0, euler_angles_0, phase_volumes_0, grain_volumes_0 = ccb_c.calc_grain_prop(M, grain_ids_0, trunc_triangles)
surface_voxels_0, gb_voxels_0, interface_voxels_0 = ccb_c.calc_surface_prop(M, grain_ids_0)

vol_frac_WC_0 = phase_volumes_0[1]/np.float(np.sum(phase_volumes_0))
vol_frac_Co_0 = 1 - vol_frac_WC_0
mass_frac_WC_0 = ccb.mass_fraction(vol_frac_WC_0)
d_eq_0 = ccb.volume_to_eq_d(grain_volumes_0*delta_x**3)

sum_gb_voxels_0 = np.sum(gb_voxels_0)
contiguity_0 = sum_gb_voxels_0 / np.float(sum_gb_voxels_0 + np.sum(interface_voxels_0))

ccb.write_hdf5('testfile_0.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_0, phases_0, good_voxels_0, euler_angles_0, surface_voxels_0, gb_voxels_0, interface_voxels_0, overlaps_0)

# Compute actual volume fraction:
print "generated volume fraction of Co (before tweaks):", vol_frac_Co_0

# Make new copies to play with the unlimited monte carle potts simulation.
if False:
    grain_ids_1 = grain_ids_0.copy()
    gb_voxels_1 = gb_voxels_0.copy()
    
    start_time = time.time()
    ccb_c.make_mcp_unlim(M, grain_ids_1, gb_voxels_1, mc_steps, kBT)
    print np.str(time.time() - start_time) + " seconds"
    
    surface_voxels_1, gb_voxels_1, interface_voxels_1 = ccb_c.calc_surface_prop(M, grain_ids_1)
    phases_1, good_voxels_1, euler_angles_1, phase_volumes_1, grain_volumes_1 = ccb_c.calc_grain_prop(M, grain_ids_1, trunc_triangles)
    
    vol_frac_WC_1 = phase_volumes_1[1]/np.float(M**3)
    vol_frac_Co_1 = 1 - vol_frac_WC_1
    mass_frac_WC_1 = ccb.mass_fraction(vol_frac_WC_1)
    d_eq_1 = ccb.volume_to_eq_d(grain_volumes_1*delta_x**3)
    
    sum_gb_voxels_1 = np.sum(gb_voxels_1)
    contiguity_1 = sum_gb_voxels_1 / np.float(sum_gb_voxels_1 + np.sum(interface_voxels_1))
    
    ccb.write_hdf5('testfile_1.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_1, phases_1, good_voxels_1, euler_angles_1, surface_voxels_1, gb_voxels_1, interface_voxels_1, overlaps_0)
    
# Make new copies to play with the bounded monte carlo potts simulation.
grain_ids_2 = grain_ids_0.copy()
gb_voxels_2 = gb_voxels_0.copy()

start_time = time.time()
ccb_c.make_mcp_bound(M, grain_ids_2, gb_voxels_2, voxel_indices_0, mc_steps, kBT)
print np.str(time.time() - start_time) + " seconds"
    
surface_voxels_2, gb_voxels_2_1, interface_voxels_2 = ccb_c.calc_surface_prop(M, grain_ids_2)
phases_2, good_voxels_2, euler_angles_2, phase_volumes_2, grain_volumes_2 = ccb_c.calc_grain_prop(M, grain_ids_2, trunc_triangles)
    
vol_frac_WC_2 = phase_volumes_2[1]/np.float(M**3)
vol_frac_Co_2 = 1 - vol_frac_WC_2
mass_frac_WC_2 = ccb.mass_fraction(vol_frac_WC_2)
d_eq_2 = ccb.volume_to_eq_d(grain_volumes_2*delta_x**3)
    
sum_gb_voxels_2 = np.sum(gb_voxels_2)
contiguity_2 = sum_gb_voxels_2 / np.float(sum_gb_voxels_2 + np.sum(interface_voxels_2))

ccb.write_hdf5('testfile_2.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_2, phases_2, good_voxels_2, euler_angles_2, surface_voxels_2, gb_voxels_2, interface_voxels_2, overlaps_0)
ccb.write_oofem('testfile_2', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_2)


# Make new copies to play with the bounded monte carlo potts simulation with stray voxel cleanup.
# No MCP_Bound is actually run here?
if False:
    grain_ids_3 = grain_ids_2.copy()
    gb_voxels_3 = gb_voxels_2.copy()

    start_time = time.time()
    ccb_c.stray_cleanup(M, grain_ids_3)
    print np.str(time.time() - start_time) + " seconds"

    surface_voxels_3, gb_voxels_3_1, interface_voxels_3 = ccb_c.calc_surface_prop(M, grain_ids_3)
    phases_3, good_voxels_3, euler_angles_3, phase_volumes_3, grain_volumes_3 = ccb_c.calc_grain_prop(M, grain_ids_3, trunc_triangles)
    
    vol_frac_WC_3 = phase_volumes_3[1]/np.float(M**3)
    vol_frac_Co_3 = 1 - vol_frac_WC_3
    mass_frac_WC_3 = ccb.mass_fraction(vol_frac_WC_3)
    d_eq_3 = ccb.volume_to_eq_d(grain_volumes_3*delta_x**3)

    sum_gb_voxels_3 = np.sum(gb_voxels_3)
    contiguity_3 = sum_gb_voxels_3 / np.float(sum_gb_voxels_3 + np.sum(interface_voxels_3))

    ccb.write_hdf5('testfile_3.hdf5', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_3, phases_3, good_voxels_3, euler_angles_3, surface_voxels_3, gb_voxels_3, interface_voxels_3, overlaps_0)
    ccb.write_oofem('testfile_3', 3*[M], 3*[delta_x], trunc_triangles, grain_ids_2)

    # Compute actual volume fraction:
    print "generated volume fraction of Co (after tweaks):", vol_frac_Co_3
    #print grain_fraction

# Misorientation:
#nbrList = findNeighbors(trunc_triangles,L)
#angles_001 = compute_all_misorientation_001(trunc_triangles,nbrList)
#print angles_001
#angles_net = compute_all_misorientation_net(trunc_triangles,nbrList)

angles, areas = compute_all_misorientation_voxel(trunc_triangles, grain_ids_2, [M]*3)

all_angles = []
for grains, angle in angles.iteritems():
	area = areas[grains]
	all_angles.extend(area * [angle])

if all_angles:
	num_bins = 40
	plt.rcParams.update({'font.size': 20})
	fig=plt.figure(num=None, figsize=(12, 7), facecolor='w', edgecolor='k')
	n, bins, patches = plt.hist(all_angles, num_bins, normed=1, facecolor='blue', alpha=0.5)
	plt.xlabel('Angle')
	plt.ylabel('Probability')
	plt.show()

