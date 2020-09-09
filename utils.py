import numpy as np


def rot_matrix_zyx(phi: float, theta: float, psi: float):
	r = np.array([
		[np.cos(psi) * np.cos(theta), -np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi),
		 np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi) * np.sin(theta)],
		[np.sin(psi) * np.cos(theta), np.cos(psi) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psi),
		 -np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi) * np.cos(phi)],
		[-np.sin(theta), np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi)]
	])
	return r

def angular_transformation_matrix_zyx(phi: float, theta: float):
	t = np.array([
		[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
		[0, np.cos(phi), -np.sin(phi)],
		[0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
	])
	return t