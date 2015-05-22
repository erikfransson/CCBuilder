CCBuilder
=========

Introduction
-------------
This is a short description of the effort to create a computer model of a three-dimensional (3D) microstructure resembling typical microstructures of fully-dense conventionally sintered WC-Co cemented carbides. The model is supposed to be used in finite-element method (FEM) modeling of WC-Co with cohesive zone models for WC/WC and WC/Co boundaries. The aim is to formulate a model that adequately reproduces the most important microstructural parameters in the WC-Co system. These include the volume fractions of the respective phases,  representative WC grain shapes,  WC grain size distributions, and the contiguity of the carbide phase.


Dream3D is a promising piece of software as it includes e.~g. functionality for surface meshing. Therefore, the output data structure of CCBuilder is made fully compatible with Dream3D data files to allow importing data into Dream3D for further processing.


How to build
-------------
To build CCbuilder run 

`python setup.py build_ext --inplace`

 and then you should be able to run the example

`python make_cc.py`



Work flow
----------
There is a number of input parameters thats need to be set before.

* Volume fraction goal, vol_frac_goal
* System size, L
* Grid size, M
* Populate voxel parameters
  * Number of trials for placing a grain onto the grid
  * Amplitude of displacement when finding optimal place in grid
* Monte Carlo parameters
  * Number of MC steps
  * Effective temperature kbT

Next the grains are created by

`prepare_triangles(vol_frac_goal, L)`

where the position, size, shape and orientation of each grain is drawn from random distributions.
