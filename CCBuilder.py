import numpy as np
import h5py
import GeometryTools
import TruncatedTriangle as tt
import scipy.optimize
import CCBuilder_c as ccb_c

def prepare_triangles(vol_frac_goal, L):
	print "Prepare triangles"
	
	r_min = 0.1; r_max = 0.4
	k_min = 0.2; k_max = 0.6
	
	d_eq_min = 0.5
	d_eq_max = 2
	
	trunc_triangles = []
	
	total_volume = L**3
	volume = 0.
	
	# Make random grains
	while volume/total_volume < vol_frac_goal:
		midpoint = L*np.random.random(3)
		r = (r_max-r_min)*np.random.random() + r_min
		k = (k_max-k_min)*np.random.random() + k_min
		d_eq = (d_eq_max-d_eq_min)*np.random.random() + d_eq_min
		
		rot_matrix = GeometryTools.random_rotation()
		
		trunc_triangle = tt.TruncatedTriangle(midpoint, rot_matrix, r, k, d_eq)
		
		trunc_triangles.append(trunc_triangle)
		
		volume += trunc_triangle.volume
	
	# Sort triangles w.r.t. volume, so that large triangles are added to the box first
	trunc_triangles.sort(key=lambda m: m.volume, reverse=True)
	
	return trunc_triangles

def make_neighbors(L, trunc_triangles, optimize=True):
	N = len(trunc_triangles)
	if optimize:
		optimize_midpoints(L, trunc_triangles)
	
	trunc_triangles_list = []
	neighbors = []
	
	for i in range(N):
		trunc_triangles_i = []
		trunc_triangles_i.append(trunc_triangles[i])
		trunc_triangles_i.extend(trunc_triangles[i].find_periodic_copies(L))
		_update_neighbors(trunc_triangles_i, trunc_triangles_list, neighbors)
		trunc_triangles_list.append(trunc_triangles_i)
	
	return trunc_triangles_list, neighbors

def optimize_midpoints(L, trunc_triangles):
	N = len(trunc_triangles)
	
	print "Optimizing midpoints of " + np.str(N) + " grains."
	
	x0 = np.ndarray.flatten(np.array([trunc_triangle.midpoint for trunc_triangle in trunc_triangles]))
	circumcircles = np.array([trunc_triangle.circumcircle for trunc_triangle in trunc_triangles])
	volumes = np.array([trunc_triangle.volume for trunc_triangle in trunc_triangles])
	
	result = scipy.optimize.minimize(ccb_c.sum_potential3_and_grad, x0, args=(L, circumcircles, volumes), method='BFGS', jac=True, tol=1E-2, options = {'disp' : True})
	
	for i in range(N):
		trunc_triangles[i].set_midpoint(np.mod(result.x[3*i:3*i+3], L))

def _update_neighbors(trunc_triangles, trunc_triangles_list, neighbors):
	nb = []
	for i, trunc_triangle_i in enumerate(trunc_triangles):
		nb_i = []
		if len(trunc_triangles_list) > 0:
			for j, trunc_triangles_j in enumerate(trunc_triangles_list):
				for k, trunc_triangle_k in enumerate(trunc_triangles_j):
					if np.sum(np.power(trunc_triangle_k.midpoint - trunc_triangle_i.midpoint, 2)) < np.power(trunc_triangle_k.circumcircle + trunc_triangle_i.circumcircle, 2):
						nb_i.append((j,k))
		nb.append(nb_i)
	neighbors.append(nb)

def mass_fraction(vol_frac):
	density_WC = 15.63 # g/cm3
	density_Co = 8.90 # g/cm3
	return vol_frac*density_WC / (vol_frac*density_WC + (1-vol_frac)*density_Co)

def volume_to_eq_d(volume):
	return 2*np.power(3*volume/(4*np.pi), 1/3.)

def write_hdf5(filename, M, spacing, trunc_triangles, grain_ids, phases, good_voxels, euler_angles, surface_voxels=None, gb_voxels=None, interface_voxels=None, overlaps=None):
	f = h5py.File(filename, "w")
	
	grp_voxel_data = f.create_group("VoxelDataContainer")
	grp_voxel_data.attrs.create("NUM_POINTS", np.array([M[0]*M[1]*M[2]]), shape=(1,), dtype='int64')
	grp_voxel_data.attrs.create("VTK_DATA_OBJECT", "VTK_STRUCTURED_POINTS", shape=(1,), dtype='S22')
	
	dset_dimensions = grp_voxel_data.create_dataset("DIMENSIONS", (3,), dtype='int64')
	dset_dimensions[...] = M
	
	dset_origin = grp_voxel_data.create_dataset("ORIGIN", (3,), dtype='float32')
	dset_origin[...] = np.array([0, 0, 0])
	
	dset_origin = grp_voxel_data.create_dataset("SPACING", (3,), dtype='float32')
	dset_origin[...] = spacing
	
	grp_cell_data = grp_voxel_data.create_group("CELL_DATA")
	
	dset_grain_ids = grp_cell_data.create_dataset("GrainIds", (M[0]*M[1]*M[2],), dtype='int32')
	dset_grain_ids.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_grain_ids.attrs.create("ObjectType", "DataArray<int32_t>", shape=(1,), dtype='S19')
	dset_grain_ids[...] = grain_ids
	
	dset_phases = grp_cell_data.create_dataset("Phases", (M[0]*M[1]*M[2],), dtype='int32')
	dset_phases.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_phases.attrs.create("ObjectType", "DataArray<int32_t>", shape=(1,), dtype='S19')
	dset_phases[...] = phases
	
	dset_good_voxels = grp_cell_data.create_dataset("GoodVoxels", (M[0]*M[1]*M[2],), dtype='uint8')
	dset_good_voxels.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_good_voxels.attrs.create("ObjectType", "DataArray<bool>", shape=(1,), dtype='S16')
	dset_good_voxels[...] = good_voxels
	
	dset_euler_angles = grp_cell_data.create_dataset("EulerAngles", (M[0]*M[1]*M[2],3), dtype='float32')
	dset_euler_angles.attrs.create("NumComponents", np.array([3]), shape=(1,), dtype='int32')
	dset_euler_angles.attrs.create("ObjectType", "DataArray<float>", shape=(1,), dtype='S17')
	dset_euler_angles[...] = euler_angles
	
	grp_ensemble_data = grp_voxel_data.create_group("ENSEMBLE_DATA")
	grp_ensemble_data.attrs.create("Name", "EnsembleData", shape=(1,), dtype='S13')
	
	dset_crystal_structs = grp_ensemble_data.create_dataset("CrystalStructures", (3,), dtype='uint32')
	dset_crystal_structs.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_crystal_structs.attrs.create("ObjectType", "DataArray<uint32_t>", shape=(1,), dtype='S20')
	dset_crystal_structs[...] = np.array([999, 1, 0])
	
	dset_phase_types = grp_ensemble_data.create_dataset("PhaseTypes", (3,), dtype='uint32')
	dset_phase_types.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_phase_types.attrs.create("ObjectType", "DataArray<uint32_t>", shape=(1,), dtype='S20')
	dset_phase_types[...] = np.array([999, 3, 1])
	
	grp_field_data = grp_voxel_data.create_group("FIELD_DATA")
	grp_field_data.attrs.create("Name", "FieldData", shape=(1,), dtype='S10')
	
	phases2 = np.zeros(len(trunc_triangles)+2, dtype='int32')
	phases2[1] = 1
	phases2[2:] = 2
	dset_phases2 = grp_field_data.create_dataset("Phases", (len(phases2),), dtype='int32')
	dset_phases2.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_phases2.attrs.create("ObjectType", "DataArray<int32_t>", shape=(1,), dtype='S19')
	dset_phases2[...] = phases2
	
	active = np.ones(len(trunc_triangles)+2, dtype='uint8')
	active[0] = 0
	dset_active = grp_field_data.create_dataset("Active", (len(active),), dtype='uint8')
	dset_active.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
	dset_active.attrs.create("ObjectType", "DataArray<bool>", shape=(1,), dtype='S16')
	dset_active[...] = active
	
	euler_angles_t = np.array([trunc_triangle.euler_angles for trunc_triangle in trunc_triangles])
	euler_angles2 = np.zeros((len(euler_angles_t)+2, 3))
	euler_angles2[2:] = euler_angles_t
	dset_euler_angles2 = grp_field_data.create_dataset("EulerAngles", (len(euler_angles2), 3), dtype='float32')
	dset_euler_angles2.attrs.create("NumComponents", np.array([3]), shape=(1,), dtype='int32')
	dset_euler_angles2.attrs.create("ObjectType", "DataArray<float>", shape=(1,), dtype='S17')
	dset_euler_angles2[...] = euler_angles2
	
	if surface_voxels != None:
		dset_surface_voxels = grp_cell_data.create_dataset("SurfaceVoxels", (M[0]*M[1]*M[2],), dtype='int8')
		dset_surface_voxels.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
		dset_surface_voxels.attrs.create("ObjectType", "DataArray<int8_t>", shape=(1,), dtype='S18')
		dset_surface_voxels[...] = surface_voxels
	
	if gb_voxels != None:
		dset_gb_voxels = grp_cell_data.create_dataset("GrainBoundaryVoxels", (M[0]*M[1]*M[2],), dtype='int8')
		dset_gb_voxels.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
		dset_gb_voxels.attrs.create("ObjectType", "DataArray<int8_t>", shape=(1,), dtype='S18')
		dset_gb_voxels[...] = gb_voxels
	
	if interface_voxels != None:
		dset_interface_voxels = grp_cell_data.create_dataset("InterfaceVoxels", (M[0]*M[1]*M[2],), dtype='int8')
		dset_interface_voxels.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
		dset_interface_voxels.attrs.create("ObjectType", "DataArray<int8_t>", shape=(1,), dtype='S18')
		dset_interface_voxels[...] = interface_voxels
	
	if overlaps != None:
		dset_overlaps = grp_cell_data.create_dataset("Overlaps", (M[0]*M[1]*M[2],), dtype='int8')
		dset_overlaps.attrs.create("NumComponents", np.array([1]), shape=(1,), dtype='int32')
		dset_overlaps.attrs.create("ObjectType", "DataArray<int8_t>", shape=(1,), dtype='S18')
		dset_overlaps[...] = overlaps
	
	f.close()
	
	filename_xdmf = filename.split('.')[0] + ".xdmf"
	write_xdmf(filename_xdmf, M, spacing, surface_voxels != None, gb_voxels != None, interface_voxels != None, overlaps != None)

def write_xdmf(filename, M, spacing, surface_voxels=False, gb_voxels=False, interface_voxels=False, overlaps=False):
	with open(filename, 'w') as f:
		filename_hdf5 = filename.split('.')[0] + ".hdf5"
		M_plus1 = [M[0]+1, M[1]+1, M[2]+1]
		
		f.write('<?xml version="1.0"?>\n<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd"[]>\n<Xdmf xmlns:xi="http://www.w3.org/2003/XInclude" Version="2.2">\n')
		f.write(' <Domain>\n\n')
		f.write('  <Grid Name="Cell Data" GridType="Uniform">\n')
		f.write('    <Topology TopologyType="3DCoRectMesh" Dimensions="{} {} {} "></Topology>\n'.format(*M_plus1))
		f.write('    <Geometry Type="ORIGIN_DXDYDZ">\n')
		f.write('      <!-- Origin -->\n')
		f.write('      <DataItem Format="XML" Dimensions="3">0 0 0</DataItem>\n')
		f.write('      <!-- DxDyDz (Spacing/Resolution)-->\n')
		f.write('      <DataItem Format="XML" Dimensions="3">{:f} {:f} {:f}</DataItem>\n'.format(*spacing))
		f.write('    </Geometry>\n')
		f.write('    <Attribute Name="EulerAngles (Cell)" AttributeType="Vector" Center="Cell">\n')
		f.write('      <DataItem Format="HDF" Dimensions="{} {} {} 3 " NumberType="Float" Precision="4" >\n'.format(*M))
		f.write('        {}:/VoxelDataContainer/CELL_DATA/EulerAngles\n'.format(filename_hdf5))
		f.write('      </DataItem>\n')
		f.write('    </Attribute>\n')
		f.write('\n')
		f.write('    <Attribute Name="GoodVoxels (Cell)" AttributeType="Scalar" Center="Cell">\n')
		f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="uchar" Precision="1" >\n'.format(*M))
		f.write('        {}:/VoxelDataContainer/CELL_DATA/GoodVoxels\n'.format(filename_hdf5))
		f.write('      </DataItem>\n')
		f.write('    </Attribute>\n')
		f.write('\n')
		f.write('    <Attribute Name="GrainIds (Cell)" AttributeType="Scalar" Center="Cell">\n')
		f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="Int" Precision="4" >\n'.format(*M))
		f.write('        {}:/VoxelDataContainer/CELL_DATA/GrainIds\n'.format(filename_hdf5))
		f.write('      </DataItem>\n')
		f.write('    </Attribute>\n')
		f.write('\n')
		f.write('    <Attribute Name="Phases (Cell)" AttributeType="Scalar" Center="Cell">\n')
		f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="Int" Precision="4" >\n'.format(*M))
		f.write('        {}:/VoxelDataContainer/CELL_DATA/Phases\n'.format(filename_hdf5))
		f.write('      </DataItem>\n')
		f.write('    </Attribute>\n')
		f.write('\n')
		if surface_voxels:
			f.write('    <Attribute Name="SurfaceVoxels (Cell)" AttributeType="Scalar" Center="Cell">\n')
			f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="Char" Precision="1" >\n'.format(*M))
			f.write('        {}:/VoxelDataContainer/CELL_DATA/SurfaceVoxels\n'.format(filename_hdf5))
			f.write('      </DataItem>\n')
			f.write('    </Attribute>\n')
			f.write('\n')
		if gb_voxels:
			f.write('    <Attribute Name="GrainBoundaryVoxels (Cell)" AttributeType="Scalar" Center="Cell">\n')
			f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="Char" Precision="1" >\n'.format(*M))
			f.write('        {}:/VoxelDataContainer/CELL_DATA/GrainBoundaryVoxels\n'.format(filename_hdf5))
			f.write('      </DataItem>\n')
			f.write('    </Attribute>\n')
			f.write('\n')
		if interface_voxels:
			f.write('    <Attribute Name="InterfaceVoxels (Cell)" AttributeType="Scalar" Center="Cell">\n')
			f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="Char" Precision="1" >\n'.format(*M))
			f.write('        {}:/VoxelDataContainer/CELL_DATA/InterfaceVoxels\n'.format(filename_hdf5))
			f.write('      </DataItem>\n')
			f.write('    </Attribute>\n')
			f.write('\n')
		if overlaps:
			f.write('    <Attribute Name="Overlaps (Cell)" AttributeType="Scalar" Center="Cell">\n')
			f.write('      <DataItem Format="HDF" Dimensions="{} {} {} " NumberType="Char" Precision="1" >\n'.format(*M))
			f.write('        {}:/VoxelDataContainer/CELL_DATA/Overlaps\n'.format(filename_hdf5))
			f.write('      </DataItem>\n')
			f.write('    </Attribute>\n')
			f.write('\n')
		f.write('  </Grid>\n')
		f.write('    <!-- *************** END OF Cell Data *************** -->\n')
		f.write('\n')
		f.write(' </Domain>\n')
		f.write('</Xdmf>\n')
		f.close()
