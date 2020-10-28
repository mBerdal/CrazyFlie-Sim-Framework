from sensor.lidar_sensor import LidarSensor
from sensor.range_sensor import RangeSensor
from drone_swarm.drone.drone import Drone
from logger.log_entry import LogEntry
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx

from matplotlib.patches import Circle

import numpy as np
from math import pi
from copy import deepcopy

class CrazyFlie(Drone):

  def __init__(self, id,  state: np.ndarray, **kwargs) -> None:
    max_range_sensor = kwargs.get('max_range_sensor', 4)
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

    super().__init__(id, state, sensors, state_noise_generator)
    self.prev_state = state

    self.state_dot = np.zeros([6,1])
    self.prev_state_dot = np.zeros([6,1])

    self.command = np.zeros([6, 1])

    acc_limits_upper_std = np.array([1, 1, 1, np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]).reshape(6, 1)
    acc_limits_lower_std = -acc_limits_upper_std

    self.acc_limits_lower = kwargs.get("acc_limits_lower", acc_limits_lower_std)
    self.acc_limits_upper = kwargs.get("acc_limits_upper", acc_limits_upper_std)

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
    rot_body_to_ned = rot_matrix_zyx(self.state[3], self.state[4], self.state[5])
    rays = rot_body_to_ned @ self.rays
    orgins = rot_body_to_ned @ self.orgins + self.state[0:3]
    return rays, orgins, self.max_ranges, self.min_ranges

  def get_sensor_limits(self):
    return self.max_ranges, self.min_ranges

  def get_sensor_idx(self):
    return self.sensor_idx

  def update_state(self, step_length):
    for msg in self.get_msgs(step_length):
      if msg.data_type == "command":
        self.command = msg.data

    self.update_state_dot(step_length)
    R = rot_matrix_zyx(self.state[3],self.state[4],self.state[5])
    T = angular_transformation_matrix_zyx(self.state[3],self.state[4])

    self.state[0:3] = self.state[0:3] + step_length * (R @ self.state_dot[0:3])
    self.state[3:6] = unwrap(self.state[3:6] + step_length * (T @ self.state_dot[3:6]))

  def update_state_dot(self, step_length):
    self.state_dot = self.state_dot + step_length*clip(self.command,self.acc_limits_lower,self.acc_limits_upper)

  def read_sensors(self, environment, vector_format=True):
    readings = []
    for s in self.sensors:
      if vector_format:
        readings.append(s.get_reading(environment.get_objects(),self.state))
      else:
        readings.append(np.linalg.norm(s.get_reading(environment.get_objects(),self.state)).item())
    return readings


  @staticmethod
  def init_plot(axis, state, **kwargs):
    c = Circle(state[0:2], radius=0.01, color="green")
    axis.add_patch(c)
    return c

  @staticmethod
  def update_plot(fig, state, **kwargs):
    fig.set_center(state[0:2])

class CrazyFlieLidar(Drone):

  def __init__(self, id,  state: np.ndarray, **kwargs) -> None:
    max_range_sensor = kwargs.get('max_range_sensor', 4)
    num_rays_sensor = kwargs.get("num_beams_sensor", 36)
    state_noise_generator = kwargs.get("state_noise_generator", lambda: np.zeros(state.shape))
    sensors = [ LidarSensor(np.array([0,0,0.01]).reshape(3,1),np.array([0,0,0]).reshape(3,1),max_range=max_range_sensor,num_rays=num_rays_sensor)
    ]

    super().__init__(id, state, sensors, state_noise_generator)
    self.prev_state = state

    self.state_dot = np.zeros([6,1])
    self.prev_state_dot = np.zeros([6,1])

    self.command = np.zeros([6, 1])

    acc_limits_upper_std = np.array([2, 2, 2, np.deg2rad(270), np.deg2rad(270), np.deg2rad(270)]).reshape(6, 1)
    acc_limits_lower_std = -acc_limits_upper_std

    self.acc_limits_lower = kwargs.get("acc_limits_lower", acc_limits_lower_std)
    self.acc_limits_upper = kwargs.get("acc_limits_upper", acc_limits_upper_std)

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
    rot_body_to_ned = rot_matrix_zyx(self.state[3], self.state[4], self.state[5])
    rays = rot_body_to_ned @ self.rays
    orgins = rot_body_to_ned @ self.orgins + self.state[0:3]
    return rays, orgins, self.max_ranges, self.min_ranges

  def get_sensor_limits(self):
    return self.max_ranges, self.min_ranges

  def get_sensor_idx(self):
    return self.sensor_idx

  def update_state(self, step_length):
    for msg in self.get_msgs(step_length):
      if msg.data_type == "command":
        self.command = msg.data

    self.update_state_dot(step_length)
    R = rot_matrix_zyx(self.state[3],self.state[4],self.state[5])
    T = angular_transformation_matrix_zyx(self.state[3],self.state[4])

    self.state[0:3] = self.state[0:3] + step_length * (R @ self.state_dot[0:3])
    self.state[3:6] = unwrap(self.state[3:6] + step_length * (T @ self.state_dot[3:6]))

  def update_state_dot(self, step_length):
    self.state_dot = self.state_dot + step_length*clip(self.command,self.acc_limits_lower,self.acc_limits_upper)

  def read_sensors(self, environment, vector_format=True):
    readings = []
    for s in self.sensors:
      if vector_format:
        readings.append(s.get_reading(environment.get_objects(),self.state))
      else:
        readings.append(np.linalg.norm(s.get_reading(environment.get_objects(),self.state)).item())
    return readings


  @staticmethod
  def init_plot(axis, state, **kwargs):
    c = Circle(state[0:2], radius=0.1, color="green")
    axis.add_patch(c)
    return c

  @staticmethod
  def update_plot(fig, state, **kwargs):
    fig.set_center(state[0:2])




def unwrap(angles):
  angles[0] = angles[0] % (2*pi) if angles[0] > 2*pi or angles[0] < 0 else angles[0]
  angles[1] = angles[1] % (2*pi) if angles[1] > 2*pi or angles[1] < 0 else angles[1]
  angles[2] = angles[2] % (2*pi) if angles[2] > 2*pi or angles[2] < 0 else angles[2]
  return angles

def clip(array, lower_bound, upper_bound):
  array[0] = max(min(array[0], upper_bound[0]), lower_bound[0])
  array[1] = max(min(array[1], upper_bound[1]), lower_bound[1])
  array[2] = max(min(array[2], upper_bound[2]), lower_bound[2])
  array[3] = max(min(array[3], upper_bound[3]), lower_bound[3])
  array[4] = max(min(array[4], upper_bound[4]), lower_bound[4])
  array[5] = max(min(array[5], upper_bound[5]), lower_bound[5])
  return array
