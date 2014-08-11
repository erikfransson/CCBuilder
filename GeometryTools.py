import numpy as np
import numpy.linalg

def cross_product_matrix(u):
	return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

def rot_matrix(u, theta):
	d = u / np.linalg.norm(u)
	
	return np.eye(3)*np.cos(theta) + np.sin(theta)*cross_product_matrix(d) + (1-np.cos(theta))*np.outer(d,d)

def rotmatrix_to_axisangle(R):
	puny = 1E-11
	
	cos_theta = 0.5*(R[0,0] + R[1,1] + R[2,2] - 1)
	sin_theta = np.sqrt(1 - cos_theta**2)
	theta = np.arccos(cos_theta)
	
	if np.abs(sin_theta) > puny:
		r0 = (R[2,1] - R[1,2])/(2*sin_theta)
		r1 = (R[0,2] - R[2,0])/(2*sin_theta)
		r2 = (R[1,0] - R[0,1])/(2*sin_theta)
	elif cos_theta > 0:
		r0 = 1.; r1 = 0.; r2 = 0.
	else:
		scr = R.copy()
		for k in range(0,3):
			scr[k,k] = scr[k,k] + 1
			n = scr[0,k]**2 + scr[1,k]**2 + scr[2,k]**2
			if n > 0:
				n = np.sqrt(n)
				r0 = scr[0,k] / n; r1 = scr[1,k] / n; r2 = scr[2,k] / n
				break
	
	return theta, np.array([r0,r1,r2])

def rotmatrix_to_euler(R):
	puny = 1E-11
	
	cos_theta = R[2,2]
	sin_theta_sq = 1 - cos_theta**2
	
	if sin_theta_sq < puny:
		theta = np.arccos(cos_theta)
		psi = 0.
		phi = np.mod(np.arctan2(R[1,0], R[0,0]), 2*np.pi)
		return np.array([phi, theta, psi])
	
	dum00 = R[0,0] + (R[1,2]*R[2,1] + R[0,2]*R[2,0]*R[2,2]) / sin_theta_sq
	dum01 = R[0,1] - (R[1,2]*R[2,0] - R[0,2]*R[2,1]*R[2,2]) / sin_theta_sq
	dum10 = R[1,0] - (R[0,2]*R[2,1] - R[1,2]*R[2,0]*R[2,2]) / sin_theta_sq
	dum11 = R[1,1] + (R[0,2]*R[2,0] + R[1,2]*R[2,1]*R[2,2]) / sin_theta_sq
	
	dum = np.sqrt(dum00**2 + dum01**2 + dum10**2 + dum11**2)
	if dum > puny:
		raise ValueError
	
	theta = np.arccos(cos_theta)
	phi = np.mod(np.arctan2(R[0,2], -R[1,2]), 2*np.pi)
	psi = np.mod(np.arctan2(R[2,0], R[2,1]), 2*np.pi)
	
	return np.array([phi, theta, psi])

def random_axis():
	mu = 2.
	
	while mu > 1:
		r = 2*np.random.random(3) - 1
		mu = np.linalg.norm(r)
	
	return r / mu

def random_rotation():
	small = 0.1
	
	r0 = random_axis()
	
	dum = 0
	while dum < small:
		r1 = random_axis()
		r1 = r1 - np.dot(r1,r0)*r0
		dum = np.linalg.norm(r1)
	
	r1 /= dum
	
	r2 = np.cross(r0, r1)
	
	return np.array([r0,r1,r2]).transpose()
		

#def misorientation2(R):
	#w,v = np.linalg.eig(R)

	#cos_theta = 0.5*(np.trace(R) - 1)

	##direction = np.real(v[:,0])
	
	#return cos_theta, v

from fractions import gcd
from fractions import Fraction

def convert(miller_vector):
	U = miller_vector[0]
	V = miller_vector[1]
	W = miller_vector[2]
	u = Fraction(2*U-V, 3)
	v = Fraction(-U+2*V, 3)
	t = Fraction(-U-V, 3)
	w = W
	
	max_denom = np.max([np.abs(u.denominator), np.abs(v.denominator), np.abs(t.denominator)])
	u *= max_denom
	v *= max_denom
	t *= max_denom
	w *= max_denom
	
	divisor = gcd(gcd(gcd(u,v), t), w)
	
	# Return Miller-Bravais indices
	return np.array([np.int(u), np.int(v), np.int(t), np.int(w)]) / np.int(np.abs(divisor))

