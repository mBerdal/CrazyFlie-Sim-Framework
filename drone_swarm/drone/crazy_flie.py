from sensor.range_sensor import RangeSensor
from drone_swarm.drone.drone import Drone
from logger.log_entry import LogEntry
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx, skew_sym
from eom import Eom
from quad_controller import QuadController

from matplotlib.patches import Circle

import numpy as np
from math import pi
from copy import deepcopy

class CrazyFlie(Drone):
  inerta_matrix = np.diag([8.5532*10**(-3), 8.5532*10**(-3), 1.476*10**(-2)])
  mass = 9.64*10**(-1)

  def __init__(self, id, state: np.ndarray, **kwargs) -> None:
    max_range_sensor = kwargs.get('max_range_sensor', 4)
    range_res_sensor = kwargs.get('range_res_sensor', 0.001)
    arc_angle_sensor = kwargs.get("arc_angle_sensor", np.deg2rad(27))
    num_beams_sensor = kwargs.get("num_beams_sensor", 11)
    state_noise_generator = kwargs.get("state_noise_generator", lambda: np.zeros(state.shape))
    sensors = [
      RangeSensor(
        np.array([((-1)**i)*0.01 if i == 0 or i == 2 else 0, ((-1)**i)*0.01 if i == 1 or i == 3 else 0, 0]).reshape(3, 1),
        np.array([0, 0, (np.pi/2)*i]).reshape(3, 1),
        max_range = max_range_sensor,
        num_rays=num_beams_sensor
      ) for i in range(4)
    ]
    
    def diff_eqn(t, state, forces, torques):
      R = rot_matrix_zyx(state[3], state[4], state[5])
      z = np.zeros((3, 3))
      n_dot = (np.block([
        [R, z],
        [z, angular_transformation_matrix_zyx(state[3], state[4])]
      ])@state[6:12]).reshape(6, 1)
      M_RB_inv = np.block([
        [(1/self.mass)*np.eye(3), z],
        [z, np.diag([1/self.inerta_matrix[0, 0], 1/self.inerta_matrix[1, 1], 1/self.inerta_matrix[2, 2]])]
      ])
      s = skew_sym(self.mass*state[6:9])
      C_RB = np.block([
        [z, -s],
        [-s, -skew_sym(self.inerta_matrix@state[9:12])]
      ])
      t_RB = np.concatenate((R.T@Eom.__g__ + forces, torques))
      v_dot = M_RB_inv@(-(C_RB@state[6:12]).reshape(6, 1) + t_RB)
      return np.concatenate((n_dot, v_dot)).reshape(12, )

    super().__init__(id, Eom(state, diff_eqn), sensors, state_noise_generator)
    self.controller = QuadController()
    self.command = np.zeros([6, 1])

    # Sensor parameters

    rays = []
    orgins =  []
    max_ranges = []
    min_ranges = []
    self.sensor_idx = []
    start_ind = 0
    for s in self.sensors:
      r, o, ma, mi = s.get_ray_vectors()
      rays.append(r)
      orgins.append(o)
      max_ranges.append(ma)
      min_ranges.append(mi)
      self.sensor_idx.append({"start":start_ind, "end": start_ind+r.shape[1]})
      start_ind = start_ind + r.shape[1]
    self.rays = np.concatenate(rays,axis=1)
    self.orgins = np.concatenate(orgins,axis=1)
    self.max_ranges = np.concatenate(max_ranges)
    self.min_ranges = np.concatenate(min_ranges)

  def get_sensor_rays(self):
    state = self.eom.get_state()
    rot_body_to_ned = rot_matrix_zyx(state[3], state[4], state[5])
    rays = rot_body_to_ned @ self.rays
    orgins = rot_body_to_ned @ self.orgins + state[0:3]
    return rays, orgins, self.max_ranges, self.min_ranges

  def get_sensor_limits(self):
    return self.max_ranges, self.min_ranges

  def get_sensor_idx(self):
    return self.sensor_idx

  def update_state(self, step_length):
    for msg in self.get_msgs(step_length):
      if msg.data_type == "command":
        self.command = msg.data

    forces, torques = self.controller.get_inputs(self.eom.get_state(), 0, 0, -1, 0, step_length, self.mass, self.inerta_matrix)
    self.eom.step(step_length, forces, torques)

  def read_sensors(self, environment, vector_format=True):
    readings = []
    for s in self.sensors:
      if vector_format:
        readings.append(s.get_reading(environment.get_objects(), self.eom.get_state()))
      else:
        readings.append(np.linalg.norm(s.get_reading(environment.get_objects(), self.eom.get_state())).item())
    return readings


  @staticmethod
  def init_plot(axis, state, **kwargs):
    c = Circle(state[0:2], radius=0.01, color="green")
    axis.add_patch(c)
    return c

  @staticmethod
  def update_plot(fig, state, **kwargs):
    fig.set_center(state[0:2])