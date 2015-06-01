# CCBuilder

## Introduction
This is a short description of the effort to create a computer model of a three-dimensional (3D) microstructure resembling typical microstructures of fully-dense conventionally sintered WC-Co cemented carbides. The model is supposed to be used in finite-element method (FEM) modeling of WC-Co with cohesive zone models for WC/WC and WC/Co boundaries. The aim is to formulate a model that adequately reproduces the most important microstructural parameters in the WC-Co system. These include the volume fractions of the respective phases,  representative WC grain shapes,  WC grain size distributions, and the contiguity of the carbide phase.


Dream3D is a promising piece of software as it includes e.~g. functionality for surface meshing. Therefore, the output data structure of CCBuilder is made fully compatible with Dream3D data files to allow importing data into Dream3D for further processing.


## How to build
CCBuilder is written in python with the computational expensive functions implemented in Cython.

To build CCbuilder run 

`python setup.py build_ext --inplace`

and then you should be able to run the example

`python make_cc.py`



## Work flow
The general work flow of CCBuilder is explained briefly below, assuming you imported the CCBuilder modules

`import CCBuilder as ccb`

`import CCBuilder_c as ccb_c`


### Input parameters
There are a number of input parameters that need to be set before running CCbuilder.

* Volume fraction goal, vol_frac_goal
* System size, L
* Grid size, M
* Populate voxel parameters
  * Number of trials for placing a grain onto the grid, nr_tries
  * Amplitude of displacement when finding optimal place on grid, delta
* Monte Carlo parameters
  * Number of MC steps, mc_steps
  * Effective temperature, kBT


### Generating grains and populating voxels
Here the process of placing triangular prism onto the grid is explained.
First the WC grains (truncated triangles objects) are prepared by

`ccb.prepare_triangles(vol_frac_goal, L)`

where the position, size, shape and orientation of each grain is drawn from random distributions.
Then optionally the midpoints of the grains can be optimized, ie trying to separate them as much as possible, by 

`ccb.optimize_midpoints(L, trunc_triangles)`

which leads to better packing of the grains. 
Next the voxels are populated meaning each voxel is assigned to a grain or the binder by

`grain_ids, overlaps, voxel_indices = ccb_c.populate_voxels(M, voxel_indices_xyz, nr_tries, delta)`

The algorithm makes nr_tries tries to insert a grain onto the grid at its midpoint position plus a random displacement. The position which give rise to minimum overlap with the other grains is chosen.  


### Corrections of unphysical artefacts
After the voxels are populated with grains there will be some stray voxels due to discretization errors. These voxels can be cleaned up by running

`ccb_c.stray_cleanup(M, grain_ids)`

where for instance if a voxel beloning to grain 2 and is fully surronded by voxels beloning to grain 3 then it is assigned to grain 3. 
NOTE !! Perhaps stray_cleanup is meant to be ran after the MCP.




It is possible to run a Potts model simulation of the structure using the Metropolis algorithm. The simulation is mainly done to correct for unphysical artefacts in the microstructures, most notably places where one grain protrudes into another one.

In order to run an Monte carlo potts simulation the grain boundary voxels are first found by

`surface_voxels, gb_voxels, interface_voxels = ccb_c.calc_surface_prop(M, grain_ids_1)`

the surface and interface voxels are not needed for the MCP run but are useful when analyzing the final structure (will be discussed more in Outputs and results).

In the MCP algorithm a grain boundary voxel is chosen at random and the grain_id of its neighbors are checked. The id of the neighboring WC grains are inserted into a set and the chosen voxels id is changed to a random id from the set. This change will lead to a total grain boundary area change dA and the change is accepted with a probability of min{ 1 , exp(-dA/kT )}. 

NOTE !! If dA = 0 then the probability is 0.5 ( set explicitly in the code ) , this should probably be 1.0 instead??

There are three variations of the monte carlo potts simulation available. 

* make_mcp_unlim
* make_mcp_overlap
* make_mcp_bound

The unlimited version just runs the algorithm above without any constraints. In this method there is no limit to the growth of individual grains and therefore large movements of the grain boundaries can take place.

In the overlap variation a random grain boundary voxel that is also an overlap voxel is chosen. This means grain boundaries can only change where two grains overlap. For high WC fraction this might not be very useful due to overlap being everywhere.

Another way to limit boundary movement is the bound method where grains are limited to their original shape describe by voxel_indices.



### Outputs and results
After the structure is generated you can calculate some intersting measurments of the structure such as volume fractions, grain size distribution, contiguity and misorientation distribution. Examples of this can be found in `make_cc.py`

The function `ccb.write_hdf5` writes the simulation data to an HDF5 file using the same format as Dream3D uses.
There is also `ccb.write_oofem` which writes the results in oofem format.

