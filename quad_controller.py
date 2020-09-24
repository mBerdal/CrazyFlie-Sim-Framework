import numpy as np
from math import exp
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx, skew_sym
from math import sin, cos

class QuadController():
  def __init__(self):
    self.z_gain_params =  {
      "k_1_1": 2.75*10**1,
      "k_1_2": 8.76*10**0,
      "k_2_1": 8.8*10**0,
      "k_2_2": 4.71*10**0,
      "k_3_1": 1.849*10**1,
      "k_3_2": 1.002*10**1,
      "mu_1": 3.1*10**-1,
      "mu_2": 3.6*10**-1,
      "mu_3": 9.8*10**-1,
      "alpha_1": 9.6*10**-1,
      "alpha_2": 9.7*10**-1,
      "alpha_3": 9.7*10**-1,
    }
    self.e = np.zeros((6, 1))
    self.e_int = np.zeros((6, 1))

  def get_inputs(self, X, vx_C, vy_C, z_C, psi_C , h, m, I):
    v_body = rot_matrix_zyx(X[3], X[4], X[5])@X[6:9]
    self.e[0], self.e_int[0], _ = QuadController.__get_errors(self.e[0], self.e_int[0], vx_C, v_body[0], h)
    self.e[1], self.e_int[1], _ = QuadController.__get_errors(self.e[1], self.e_int[1], vy_C, v_body[1], h)
    self.e[2], self.e_int[2], e_z_dot = QuadController.__get_errors(self.e[2], self.e_int[2], z_C, X[2], h)
    f_t = QuadController.__f(self.e[2], self.z_gain_params["k_1_1"], self.z_gain_params["k_1_2"], self.z_gain_params["mu_1"], self.z_gain_params["alpha_1"])\
      + QuadController.__f(e_z_dot, self.z_gain_params["k_2_1"], self.z_gain_params["k_2_2"], self.z_gain_params["mu_2"], self.z_gain_params["alpha_2"])\
      + QuadController.__f(self.e_int[2], self.z_gain_params["k_3_1"], self.z_gain_params["k_3_2"], self.z_gain_params["mu_3"], self.z_gain_params["alpha_3"])

    eta2_C = np.concatenate(((m/f_t)*np.array([[sin(X[5]), -cos(X[5])], [cos(X[5]), sin(X[5])]])@np.diag([1, 1])@self.e[0:2], np.array([psi_C]).reshape(1, 1)))

    self.e[3:6], self.e_int[3:6], e_eta2_dot = QuadController.__get_errors(self.e[3:6], self.e_int[3:6], eta2_C, X[3:6], h)
    mu2_C = np.linalg.inv(angular_transformation_matrix_zyx(X[3], X[4]))@(np.array([6, 6, 2.5]).reshape(3, 1)*self.e[3:6] + np.array([0, 0, 0.1]).reshape(3, 1)*self.e_int[3:6] + np.array([0, 0, 0.1]).reshape(3, 1)*e_eta2_dot)
    tau_hat = 0.002*np.linalg.inv(I)@(mu2_C - X[9:12])
    tau = -skew_sym(m*X[6:9])@X[6:9] - skew_sym(I@X[9:12])@X[9:12] + tau_hat
    return np.concatenate((np.array([0, 0]), f_t)).reshape(3, 1), tau

  @staticmethod
  def __get_errors(error, error_int, state_c, state, h):
    e_prev = error
    e = state_c - state
    e_int = error_int + h*e
    e_dot = (e - e_prev)/h
    return e, e_int, e_dot
  
  @staticmethod
  def __k(beta, k_i_1, k_i_2, mu_i):
    try:
      return k_i_1 + k_i_2/(1 + exp(mu_i*(beta**2)))
    except OverflowError:
      return k_i_1

  @staticmethod
  def __f(beta, k_i_1, k_i_2, mu_i, alpha_i):
    return QuadController.__k(beta, k_i_1, k_i_2, mu_i)*(abs(beta)**alpha_i)*np.sign(beta)