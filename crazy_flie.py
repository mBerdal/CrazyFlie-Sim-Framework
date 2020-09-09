from range_sensor import RangeSensor, ZeroRangeException, FullRangeException
from sensor import Sensor
from environment import Environment
from communication import CommunicationNode
from utils.rotation_utils import rot_matrix_zyx, angular_transformation_matrix_zyx

import numpy as np
from typing import List

class CrazyFlie(CommunicationNode):
  def __init__(self, id,  initial_state: np.ndarray, initial_state_dot: np.ndarray, sensors: List[Sensor],
               acc_limits_lower: np.ndarray, acc_limits_upper: np.ndarray) -> None:
    self.state = initial_state
    self.prev_state = initial_state
    self.state_dot = initial_state_dot
    self.prev_state_dot = initial_state_dot
    self.sensors = sensors
    self.crashed = False
    self.id = id
    self.command = initial_state_dot
    self.acc_limits_lower = acc_limits_lower
    self.acc_limits_upper = acc_limits_upper

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
    anggular = self.state[3:6] + time_step * np.matmul(T, self.state_dot[3:6].reshape(3,1))
    self.state = np.concatenate([trans,anggular]).reshape(6,1)


  def update_state_dot(self, time_step):
    self.prev_state_dot = self.state_dot
    self.state_dot = self.state_dot + time_step*np.clip(self.command-self.state_dot,self.acc_limits_lower,self.acc_limits_upper)

  def read_sensors(self, env: Environment, vector_format=True):
    readings = []
    for s in self.sensors:
      if vector_format:
        readings.append(s.get_reading(env,self.state))
      else:
        readings.append(np.linalg.norm(s.get_reading(env,self.state)).item())
    return readings


  def do_step(self, environment: Environment, time_step: float) -> None:
    assert not self.crashed
    
    avg_obstacle_vec = np.zeros((6,1))
    for s in self.sensors:
      if isinstance(s, RangeSensor):
        try:
          measurement = s.get_reading(environment, self.state)
          avg_obstacle_vec += np.concatenate([measurement.reshape(3,1),np.zeros([3,1])])
        except ZeroRangeException as zre:
          print(zre.msg)
          self.crashed = True
        except FullRangeException as fre:
          print(fre.msg)

    self.state -= avg_obstacle_vec

  def plot(self, axis, environment):
    self.fig, = axis.plot([self.state[0]], [self.state[1]], 'go')
    for s in self.sensors:
      if isinstance(s, RangeSensor):
        s.plot(axis,environment,self.state)

  def update_plot(self, environment):
    self.fig.set_data(self.state[0], self.state[1])
    for s in self.sensors:
      if isinstance(s, RangeSensor):
        s.update_plot(environment,self.state)

class CrazyFlieCrash(Exception):
  pass