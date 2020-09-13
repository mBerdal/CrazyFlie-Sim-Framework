from range_sensor import RangeSensor, ZeroRangeException, FullRangeException
from sensor import Sensor
from environment import Environment
from communication import CommunicationNode
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx
from drone import Drone

import numpy as np
from typing import List

class CrazyFlie(Drone,CommunicationNode):

  def __init__(self, id,  initial_state: np.ndarray, **kwargs) -> None:
    self.state = initial_state
    self.prev_state = initial_state

    self.state_dot = np.zeros([6,1])
    self.prev_state_dot = np.zeros([6,1])

    self.id = id
    self.command = None

    acc_limits_lower_std = -np.array([1, 1, 1, np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]).reshape(6, 1)
    acc_limits_upper_std = np.array([1, 1, 1, np.deg2rad(10), np.deg2rad(10), np.deg2rad(10)]).reshape(6, 1)

    self.acc_limits_lower = kwargs.get("acc_limits_lower", acc_limits_lower_std)
    self.acc_limits_upper = kwargs.get("acc_limits_upper", acc_limits_upper_std)

    # Sensor parameters
    max_range_sensor = kwargs.get('max_range_sensor', 4)
    range_res_sensor = kwargs.get('range_res_sensor',0.001)
    arc_angle_sensor = kwargs.get("arc_angle_sensor",np.deg2rad(27))
    num_beams_sensor = kwargs.get("num_beams_sensor",9)
    arc_res_sensor = arc_angle_sensor/num_beams_sensor

    attitude_body1 = np.array([0, 0, np.pi]).reshape(3, 1)
    pos_body1 = np.array([0.01, 0, 0]).reshape(3, 1)
    sensor1 = RangeSensor(max_range_sensor, range_res_sensor, arc_angle_sensor, arc_res_sensor, pos_body1, attitude_body1)
    attitude_body2 = np.array([0, 0, np.pi / 2]).reshape(3, 1)
    pos_body2 = np.array([0, 0.01, 0]).reshape(3, 1)
    sensor2 = RangeSensor(max_range_sensor, range_res_sensor, arc_angle_sensor, arc_res_sensor, pos_body2, attitude_body2)
    attitude_body3 = np.array([0, 0, -np.pi / 2]).reshape(3, 1)
    pos_body3 = np.array([-0.01, 0, 0]).reshape(3, 1)
    sensor3 = RangeSensor(max_range_sensor, range_res_sensor, arc_angle_sensor, arc_res_sensor, pos_body3, attitude_body3)
    attitude_body4 = np.array([0, 0, 0]).reshape(3, 1)
    pos_body4 = np.array([0, -0.01, 0]).reshape(3, 1)
    sensor4 = RangeSensor(max_range_sensor, range_res_sensor, arc_angle_sensor, arc_res_sensor, pos_body4, attitude_body4)

    self.sensors = [sensor1, sensor2, sensor3, sensor4]

  def recv_msg(self, msg):
    print(f"{self} reveiced msg")
    print(msg)

  def update_command(self, command):
    self.command = command

  def update_state(self, time_step):
    self.update_state_dot(time_step)
    self.prev_state = self.state

    R = rot_matrix_zyx(self.state.item(3),self.state.item(4),self.state.item(5))
    T = angular_transformation_matrix_zyx(self.state.item(3),self.state.item(4))

    trans = self.state[0:3] + time_step * np.matmul(R, self.state_dot[0:3].reshape(3,1))
    anggular = np.unwrap(self.state[3:6] + time_step * np.matmul(T, self.state_dot[3:6].reshape(3,1)))

    self.state = np.concatenate([trans,anggular]).reshape(6,1)


  def update_state_dot(self, time_step):
    self.prev_state_dot = self.state_dot
    self.state_dot = self.state_dot + time_step*np.clip(self.command-self.state_dot,self.acc_limits_lower,self.acc_limits_upper)

  def read_sensors(self, env: Environment, vector_format=True):
    readings = []
    for s in self.sensors:
      if vector_format:
        readings.append(s.get_reading(env.get_objects(),self.state))
      else:
        readings.append(np.linalg.norm(s.get_reading(env.get_objects(),self.state)).item())
    return readings


  def plot(self, axis, environment,plot_sensors=False):
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