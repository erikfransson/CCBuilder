import numpy as np
import GeometryTools

class TruncatedTriangle:
	def __init__(self, midpoint, rot_matrix, r, k, d_eq):
		self.r = r
		self.k = k
		self.d_eq = d_eq
		
		self.L = (1/k * 8/3. * (2*r+1)**3 / ((r**2+4*r+1)*(r+1)) * 4*np.pi/3)**(1/3.) * d_eq * 0.5
		
		self.a_long = 1/(2*r+1) * self.L
		self.a_short = r * self.a_long
	
		self.H = np.sqrt(3)/2 * self.L
		self.h = (r+1)/(2*r+1) * self.H
		
		self.t = k*self.h
		
		self.circumcircle = np.sqrt((self.a_long/2.)**2 + (self.H/3.)**2 + (self.t/2.)**2)
		
		self.rot_matrix = rot_matrix
		self.rot_matrix_tr = rot_matrix.transpose()
		angle, axis = GeometryTools.rotmatrix_to_axisangle(rot_matrix)
		self.rot_axis = axis
		self.rot_angle = angle
		self.euler_angles = GeometryTools.rotmatrix_to_euler(rot_matrix)
		
		self.basal_area = self.H*self.L/2. - 3*np.sqrt(3)/4 * self.a_short**2
		self.volume = self.basal_area*self.t
		
		# vertices in coordinates of the triangle
		if r > 0:
			self.vertices = np.array([[self.a_long/2, -self.H/3, -self.t/2], [self.a_long/2 + self.a_short/2, np.sqrt(3)/2*self.a_short - self.H/3, -self.t/2], [self.a_short/2, self.h - self.H/3, -self.t/2], [-self.a_short/2, self.h - self.H/3, -self.t/2], [-self.a_long/2 - self.a_short/2, np.sqrt(3)/2*self.a_short - self.H/3, -self.t/2], [-self.a_long/2, -self.H/3, -self.t/2]])
		else:
			self.vertices = np.array([[self.L/2, -self.H/3, -self.t/2], [0, 2*self.H/3, -self.t/2], [-self.L/2, -self.H/3, -self.t/2]])
		
		self.set_midpoint(midpoint)
	
	def set_midpoint(self, midpoint):
		self.midpoint = midpoint
		
		# vertices in external coordinates (after rotation and translation)
		self.vertices_real = []
		for vert in self.vertices:
			vert_0 = np.dot(self.rot_matrix, vert)
			vert_0 += self.midpoint
			self.vertices_real.append(vert_0)
			vert_1 = vert.copy()
			vert_1[2] = self.t/2
			vert_1 = np.dot(self.rot_matrix, vert_1)
			vert_1 += self.midpoint
			self.vertices_real.append(vert_1)
		
		self.min_x = np.min([vert[0] for vert in self.vertices_real])
		self.max_x = np.max([vert[0] for vert in self.vertices_real])
		self.min_y = np.min([vert[1] for vert in self.vertices_real])
		self.max_y = np.max([vert[1] for vert in self.vertices_real])
		self.min_z = np.min([vert[2] for vert in self.vertices_real])
		self.max_z = np.max([vert[2] for vert in self.vertices_real])
	
	@staticmethod
	def _testline(p_t, p0, p1):
		return (p_t[1] - p0[1])*(p1[0] - p0[0]) - (p1[1] - p0[1])*(p_t[0] - p0[0])
	
	# determines if the point r0 is inside the truncated triangle
	def inside(self, r0):
		r0 = r0 - self.midpoint
		
		# Rotate to coordinates of the triangle as used above
		r0 = np.dot(self.rot_matrix_tr, r0)
		
		#print r0
		
		# the triangle is within the x-y plane.
		if self.r > 0:
			return r0[2] > -self.t/2 and r0[2] < self.t/2 and TruncatedTriangle._testline(r0, self.vertices[2], self.vertices[1]) < 0 and TruncatedTriangle._testline(r0, self.vertices[3], self.vertices[2]) < 0 and TruncatedTriangle._testline(r0, self.vertices[4], self.vertices[3]) < 0 and TruncatedTriangle._testline(r0, self.vertices[4], self.vertices[5]) > 0 and TruncatedTriangle._testline(r0, self.vertices[5], self.vertices[0]) > 0 and TruncatedTriangle._testline(r0, self.vertices[0], self.vertices[1]) > 0
		else:
			return r0[2] > -self.t/2 and r0[2] < self.t/2 and TruncatedTriangle._testline(r0, self.vertices[1], self.vertices[0]) < 0 and TruncatedTriangle._testline(r0, self.vertices[2], self.vertices[1]) < 0 and TruncatedTriangle._testline(r0, self.vertices[2], self.vertices[0]) > 0
	
	def find_periodic_copies(self, L):
		border = [[self.midpoint[0] + self.circumcircle > L,
			self.midpoint[0] - self.circumcircle < 0],
			[self.midpoint[1] + self.circumcircle > L,
			self.midpoint[1] - self.circumcircle < 0],
			[self.midpoint[2] + self.circumcircle > L,
			self.midpoint[2] - self.circumcircle < 0]]
		
		transl = [np.array([L,0,0]), np.array([0,L,0]), np.array([0,0,L])]
		
		periodic_copies = []
		
		for i,on_border in enumerate(border):
			for m,border_m in enumerate(on_border):
				if border_m:
					periodic_copies.append(TruncatedTriangle(self.midpoint + (-1+2*m)*transl[i], self.rot_matrix, self.r, self.k, self.d_eq))
		
		combs = [[0,1], [0,2], [1,2]]
		for comb in combs:
			on_border_0 = border[comb[0]]
			on_border_1 = border[comb[1]]
			for m,border_m in enumerate(on_border_0):
				for n,border_n in enumerate(on_border_1):
					if border_m and border_n:
						periodic_copies.append(TruncatedTriangle(self.midpoint + (-1+2*m)*transl[comb[0]] + (-1+2*n)*transl[comb[1]], self.rot_matrix, self.r, self.k, self.d_eq))
		
		for i0,border_0 in enumerate(border[0]):
			for i1,border_1 in enumerate(border[1]):
				for i2,border_2 in enumerate(border[2]):
					if border_0 and border_1 and border_2:
						periodic_copies.append(TruncatedTriangle(self.midpoint + (-1+2*i0)*transl[0] + (-1+2*i1)*transl[1] + (-1+2*i2)*transl[2], self.rot_matrix, self.r, self.k, self.d_eq))
		
		return periodic_copies
