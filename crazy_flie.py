from range_sensor import RangeSensor, ZeroRangeException, FullRangeException
from sensor import Sensor
from environment import Environment
from communication import CommunicationNode

import numpy as np
from typing import List

class CrazyFlie(CommunicationNode):
  def __init__(self, initial_state: np.ndarray, sensors: List[Sensor]) -> None:
    self.state = initial_state
    self.sensors = sensors
    self.crashed = False

  def recv_msg(self, msg):
    print(f"{self} reveiced msg")
    print(msg)

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