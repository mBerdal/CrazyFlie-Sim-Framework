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
