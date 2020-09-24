import numpy as np
from math import sin, cos, tan

def rot_matrix_zyx(phi: float, theta: float, psi: float):
	cos_psi = cos(psi)
	sin_psi = sin(psi)
	cos_theta = cos(theta)
	sin_theta = sin(theta)
	cos_phi = cos(phi)
	sin_phi = sin(phi)
	r = np.array([
		[cos_psi * cos_theta, -sin_psi * cos_phi + cos_psi * sin_theta * sin_phi,
		 sin_psi * sin_phi + cos_psi * cos_phi * sin_theta],
		[sin_psi * cos_theta, cos_psi * cos_phi + sin_phi * sin_theta * sin_psi,
		 -cos_psi * sin_phi + sin_theta * sin_psi * cos_phi],
		[-sin_theta, cos_theta * sin_phi, cos_theta * cos_phi]
	])
	return r

def angular_transformation_matrix_zyx(phi: float, theta: float):
	sin_phi = sin(phi)
	cos_phi = cos(phi)
	tan_theta = tan(theta)
	cos_theta = cos(theta)
	t = np.array([
		[1, sin_phi*tan_theta, cos_phi*tan_theta],
		[0, cos_phi, -sin_phi],
		[0, sin_phi/cos_theta, cos_phi/cos_theta]
	])
	return t

def skew_sym(vec):
  return np.array([
    [0, -vec.item(2), vec.item(1)],
    [vec.item(2), 0, -vec.item(0)],
    [-vec.item(1), vec.item(0), 0]
  ]).reshape(3, 3)

def unwrap_angle(angle):
  return angle % 2*np.pi if angle > 2*np.pi or angle < 0 else angle

def unwrap_angles(angles):
  return np.array([unwrap_angle(angles[i]) for i in range(len(angles))])