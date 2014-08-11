import numpy as np

def make_truncated_triangle(geompy, trunc_tri):
	sk = geompy.Sketcher2D()
	sk.addPoint(trunc_tri.vertices[0][0], trunc_tri.vertices[0][1])
	if trunc_tri.r > 0:
		for i in range(1,6):
			sk.addSegmentAbsolute(trunc_tri.vertices[i][0], trunc_tri.vertices[i][1])
	else:
		for i in range(1,3):
			sk.addSegmentAbsolute(trunc_tri.vertices[i][0], trunc_tri.vertices[i][1])
	sk.close()
	
	OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)
	geomObj = geompy.MakeMarker(0, 0, 0, 1, 0, 0, 0, 1, 0)
	
	Sketch_1 = sk.wire(geomObj)
	Face_1 = geompy.MakeFaceWires([Sketch_1], 1)
	Extrusion_1 = geompy.MakePrismVecH(Face_1, OZ, trunc_tri.t)
	geompy.TranslateDXDYDZ(Extrusion_1, 0, 0, -0.5*trunc_tri.t)
	
	axis = geompy.MakeVectorDXDYDZ(trunc_tri.rot_axis[0], trunc_tri.rot_axis[1], trunc_tri.rot_axis[2])
	geompy.Rotate(Extrusion_1, axis, trunc_tri.rot_angle)
	
	geompy.TranslateDXDYDZ(Extrusion_1, trunc_tri.midpoint[0], trunc_tri.midpoint[1], trunc_tri.midpoint[2])
	
	return Extrusion_1
