CCBuilder
=========

Introduction
-------------
This is a short description of the effort to create a computer model of a three-dimensional (3D) microstructure resembling typical microstructures of fully-dense conventionally sintered WC-Co cemented carbides. The model is supposed to be used in finite-element method (FEM) modeling of WC-Co with cohesive zone models for WC/WC and WC/Co boundaries. The aim is to formulate a model that adequately reproduces the most important microstructural parameters in the WC-Co system. These include the volume fractions of the respective phases,  representative WC grain shapes,  WC grain size distributions, and the contiguity of the carbide phase.


Dream3D is a promising piece of software as it includes e.~g. functionality for surface meshing. Therefore, the output data structure of CCBuilder is made fully compatible with Dream3D data files to allow importing data into Dream3D for further processing.


How to build
-------------
CCBuilder is written in python with the computational expensive functions implemented in Cython.

To build CCbuilder run 

`python setup.py build_ext --inplace`

 and then you should be able to run the example

`python make_cc.py`



Work flow
----------
There is a number of input parameters that need to be set in order to run CCbuilder.

* Volume fraction goal, vol_frac_goal
* System size, L
* Grid size, M
* Populate voxel parameters
  * Number of trials for placing a grain onto the grid
  * Amplitude of displacement when finding optimal place in grid
* Monte Carlo parameters
  * Number of MC steps
  * Effective temperature kbT

Below the general work flow of CCBuilder is briefly discussed. 
First the WC grains (truncated triangles objects) are prepared by

`prepare_triangles(vol_frac_goal, L)`

where the position, size, shape and orientation of each grain is drawn from random distributions.

Then the midpoints of the grains are optimized, ie trying to separate them as much as possible, by 

`ccb.optimize_midpoints(L, trunc_triangles)`

which leads to a better packing of the grains. Next the voxels are populate meaning each voxel is assigned to a grain or the binder by

`voxel_indices_xyz = ccb_c.make_voxel_indices(L, M, trunc_triangles)`

`ccb_c.populate_voxels(M, voxel_indices_xyz, nr_tries, delta)`

where voxel_indices_xyz contains which voxels lies inside each grain and the call populate_voxels will place the grains onto the grid. The algorithm makes N tries to insert a grain at approximately its midpoint position. The position which give rise to minimum overlap with the other grains is chosen. 


+++ calc_surface_prop in order to otain GB_voxels for MCP run.

+++ Optional stray_cleanup, what does this function do?

+++ Monte Carlo potts simulations
+++++ The different variations of MCP, unlim bound overlap

+++calc_grain_prop to obtain some data to be written to output files
+++perhaps show how to compute vol_fracs, contiguity and misorientation given the output variables.

The function `ccb.write_hdf5` writes the simulation data to an HDF5 file using the same format as Dream3D uses.


