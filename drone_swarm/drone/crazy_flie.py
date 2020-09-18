from sensor.range_sensor import RangeSensor
from communication import CommunicationNode
from drone_swarm.drone.drone import Drone
from logger.log_entry import LogEntry
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx

import numpy as np
from math import pi
from copy import deepcopy

class CrazyFlie(Drone, CommunicationNode):

  def __init__(self, id,  state: np.ndarray, **kwargs) -> None:
    max_range_sensor = kwargs.get('max_range_sensor', 4)
    range_res_sensor = kwargs.get('range_res_sensor',0.001)
    arc_angle_sensor = kwargs.get("arc_angle_sensor",np.deg2rad(27))
    num_beams_sensor = kwargs.get("num_beams_sensor",11)
    attitude_body1 = np.array([0, 0, np.pi]).reshape(3, 1)
    pos_body1 = np.array([0.01, 0, 0]).reshape(3, 1)
    sensor1 = RangeSensor(pos_body1, attitude_body1,max_range = max_range_sensor, num_rays=num_beams_sensor)
    attitude_body2 = np.array([0, 0, np.pi / 2]).reshape(3, 1)
    pos_body2 = np.array([0, 0.01, 0]).reshape(3, 1)
    sensor2 = RangeSensor(pos_body2, attitude_body2,max_range = max_range_sensor, num_rays=num_beams_sensor)
    attitude_body3 = np.array([0, 0, -np.pi / 2]).reshape(3, 1)
    pos_body3 = np.array([-0.01, 0, 0]).reshape(3, 1)
    sensor3 = RangeSensor(pos_body3, attitude_body3,max_range = max_range_sensor, num_rays=num_beams_sensor)
    attitude_body4 = np.array([0, 0, 0]).reshape(3, 1)
    pos_body4 = np.array([0, -0.01, 0]).reshape(3, 1)
    sensor4 = RangeSensor(pos_body4, attitude_body4,max_range = max_range_sensor, num_rays=num_beams_sensor)

    super().__init__(id, state, [sensor1, sensor2, sensor3, sensor4])
    self.prev_state = state

    self.state_dot = np.zeros([6,1])
    self.prev_state_dot = np.zeros([6,1])

    self.command = None

    acc_limits_lower_std = -np.array([1, 1, 1, np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]).reshape(6, 1)
    acc_limits_upper_std = np.array([1, 1, 1, np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]).reshape(6, 1)

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
      r,o, ma, mi = s.get_ray_vectors()
      rays.append(r)
      orgins.append(o)
      max_ranges.append(ma)
      min_ranges.append(mi)
      self.sensor_idx.append({"start":start_ind,"end": start_ind+r.shape[1]})
      start_ind = start_ind + r.shape[1]
    self.rays = np.concatenate(rays,axis=1)
    self.orgins = np.concatenate(orgins,axis=1)
    self.max_ranges = np.concatenate(max_ranges)
    self.min_ranges = np.concatenate(min_ranges)

  def recv_msg(self, msg):
    print(f"{self} reveiced msg")
    print(msg)

  def get_sensor_rays(self):
    rot_body_to_ned = rot_matrix_zyx(self.state[3],self.state[4],self.state[5])
    rays = rot_body_to_ned @ self.rays
    orgins = rot_body_to_ned @ self.orgins + self.state[0:3]
    return rays, orgins, self.max_ranges, self.min_ranges

  def get_sensor_limits(self):
    return self.max_ranges, self.min_ranges

  def get_sensor_idx(self):
    return self.sensor_idx

  def update_command(self, command):
    self.command = command

  def update_state(self, time_step):
    self.update_state_dot(time_step)
    R = rot_matrix_zyx(self.state[3],self.state[4],self.state[5])
    T = angular_transformation_matrix_zyx(self.state[3],self.state[4])

    self.state[0:3] = self.state[0:3] + time_step * (R @ self.state_dot[0:3])
    self.state[3:6] = unwrap(self.state[3:6] + time_step * (T @ self.state_dot[3:6]))
    return deepcopy(self.state)

  def update_state_dot(self, time_step):
    self.state_dot = self.state_dot + time_step*clip(self.command-self.state_dot,self.acc_limits_lower,self.acc_limits_upper)

  def read_sensors(self, environment, vector_format=True):
    readings = []
    for s in self.sensors:
      if vector_format:
        readings.append(s.get_reading(environment.get_objects(),self.state))
      else:
        readings.append(np.linalg.norm(s.get_reading(environment.get_objects(),self.state)).item())
    return readings


  def plot(self, axis, environment, plot_sensors=False):
    self.fig, = axis.plot([self.state[0]], [self.state[1]], 'go')
    if plot_sensors:
      for s in self.sensors:
        if isinstance(s, RangeSensor):
          s.plot(axis,environment,self.state)

  def update_plot(self, environment,plot_sensors=False):
    self.fig.set_data(self.state[0], self.state[1])
    if plot_sensors:
      for s in self.sensors:
        if isinstance(s, RangeSensor):
          s.update_plot(environment,self.state)


def unwrap(angles):
  angles[0] = angles[0] % 2*pi if angles[0] > 2*pi or angles[0] < 0 else angles[0]
  angles[1] = angles[1] % 2*pi if angles[1] > 2*pi or angles[1] < 0 else angles[1]
  angles[2] = angles[2] % 2*pi if angles[2] > 2*pi or angles[2] < 0 else angles[2]
  return angles

def clip(array, lower_bound, upper_bound):
  array[0] = max(min(array[0], upper_bound[0]), lower_bound[0])
  array[1] = max(min(array[1], upper_bound[1]), lower_bound[1])
  array[2] = max(min(array[2], upper_bound[2]), lower_bound[2])
  array[3] = max(min(array[3], upper_bound[3]), lower_bound[3])
  array[4] = max(min(array[4], upper_bound[4]), lower_bound[4])
  array[5] = max(min(array[5], upper_bound[5]), lower_bound[5])
  return array

def test():
  state = np.array([5, 5, 0, 0, 0, 0]).reshape((6, 1))
  id = 1
  cf = CrazyFlie(id,state)
  rays, orgins = cf.get_sensor_rays()
  print(rays)
  print(orgins)