from range_sensor import RangeSensor, ZeroRangeException, FullRangeException
from sensor import Sensor
from environment import Environment
import numpy as np
from typing import List

class CrazyFlie():
  def __init__(self, initial_state: np.ndarray, sensors: List[Sensor]) -> None:
    self.state = initial_state
    self.sensors = sensors
    self.crashed = False

  def do_step(self, environment: Environment, time_step: float) -> None:
    assert not self.crashed
    
    avg_obstacle_vec = np.zeros((Environment.__SPATIAL_DIMS__,))
    for s in self.sensors:
      if isinstance(s, RangeSensor):
        try:
          dist = s.get_reading(environment, self.state, 0)
          avg_obstacle_vec += dist
        except ZeroRangeException as zre:
          print(zre.msg)
          self.crashed = True
        except FullRangeException as fre:
          print(fre.msg)

    self.state -= avg_obstacle_vec

  def plot(self, axis):
    self.fig, = axis.plot([self.state[0]], [self.state[1]], 'go')
    for s in self.sensors:
      if isinstance(s, RangeSensor): s.plot(axis, self.state, 0)

  def update_plot(self):
    self.fig.set_data(self.state[0], self.state[1])
    for s in self.sensors:
      if isinstance(s, RangeSensor): s.update_plot(self.state, 0)

class CrazyFlieCrash(Exception):
  pass