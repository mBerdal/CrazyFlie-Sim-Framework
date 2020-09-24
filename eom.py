import numpy as np
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx, skew_sym
from scipy.integrate import solve_ivp
from math import cos, sin, exp
from matplotlib import pyplot as plt
from copy import deepcopy

class Eom():

  __g__ = np.array([0, 0, 9.81]).reshape(3, 1)

  def __init__(self, X_0, diff_eqn):
    h, _ = X_0.shape
    self.X = X_0.reshape(h, )
    self.t = 0
    self.diff_eqn = diff_eqn

  def step(self, h, *args):
    s = solve_ivp(self.diff_eqn, (self.t, self.t + h), self.X, args=args)
    self.X = s.y[:, -1]
    self.t += h

  def get_state(self):
    return deepcopy(self.X).reshape(len(self.X), 1)