from sensor import Sensor
from typing import Tuple
import numpy as np
from math import cos, sin, atan
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from environment import Environment

class RangeSensor(Sensor):
  """
  Class representing a range sensor
  ...

  Attributes
  ----------
  max_range: float
    Maximum distance in meters sensor can measure
  range_res: float
    Smallest amount by which sensor can differentiate distances
  arc_angle: float
    Sensor field of view in radians
  angle_res:
    Smalles amount by which sensor can differentiate angles in radians
  self_pos_host_BODY: np.ndarray
    Position of sensor relative to host represented in the coordinate system of the host
  ang_host_BODY_self_BODY: float
    Angle of x-axis of sensors coordinate system relative to x-axis of host's coordinate system
  
  Methods
  ----------
  get_reading(environment, pos_host_NED, ang_host_NED): np.ndarray/None
    Takes as argument an environment object, position of host represented in the NED coordinate system and
    orientation of host's x-axis relative to x-axis of the NED coordinate system. If there are obstacles
    in the environment that are closer to the sensor than max_range and within field of view it returns
    a vector from the origin of host's coordinate to the nearest obstacle. If there are no obstacles closer
    than max_range and within field of view it returns None
  """

  def __init__(self, max_range: float, range_res: float , arc_angle: float, angle_res: float, self_pos_host_BODY: np.ndarray, ang_host_BODY_self_BODY: float) -> None:
    super().__init__()
    self.max_range = max_range
    self.range_res = range_res
    self.arc_angle = arc_angle
    self.angle_res = angle_res
    self.self_pos_host_BODY = self_pos_host_BODY
    self.ang_host_BODY_self_BODY = ang_host_BODY_self_BODY

  def get_reading(self, environment: Environment, pos_host_NED: np.ndarray, ang_host_NED: float) -> np.ndarray:
    assert pos_host_NED.shape == (Environment.__SPATIAL_DIMS__, ), f"host position has shape {pos_host_NED.shape}, should be {(Environment.__SPATIAL_DIMS__, )}"

    pos_NED = self.transform_to_NED_from_host_BODY(pos_host_NED, ang_host_NED, self.self_pos_host_BODY)
    ang_NED = ang_host_NED + self.ang_host_BODY_self_BODY

    obstacle_pos_host_BODY = None
    obstacle_pos_NED = None
    for r in np.arange(0, self.max_range, self.range_res):
      for ang in np.arange(-self.arc_angle/2, self.arc_angle/2 + self.angle_res, self.angle_res):
        obstacle_pos_NED = pos_NED + r*np.array([cos(ang + ang_NED), sin(ang + ang_NED)])

        if environment[obstacle_pos_NED[0], obstacle_pos_NED[1]]:
          if r == 0:
            raise ZeroRangeException()
          candidate_obstacle_pos_host_BODY = self.transform_to_host_BODY_from_NED(pos_host_NED, ang_host_NED, obstacle_pos_NED)

          if obstacle_pos_host_BODY is None:
            obstacle_pos_host_BODY = candidate_obstacle_pos_host_BODY

          elif np.linalg.norm(candidate_obstacle_pos_host_BODY) < np.linalg.norm(obstacle_pos_host_BODY):
            obstacle_pos_host_BODY = candidate_obstacle_pos_host_BODY

      if not obstacle_pos_host_BODY is None:
        return obstacle_pos_host_BODY

    raise FullRangeException()

  def transform_to_NED_from_host_BODY(self, pos_host_NED: np.ndarray, ang_host_NED: float, vec_host_BODY) -> np.ndarray:
    T_NED_host_BODY = np.array([
      [cos(ang_host_NED), -sin(ang_host_NED), pos_host_NED[0]],
      [sin(ang_host_NED), cos(ang_host_NED), pos_host_NED[1]],
      [0, 0, 1]
    ])
    return np.dot(T_NED_host_BODY, np.concatenate((vec_host_BODY, np.array([1]))))[0:Environment.__SPATIAL_DIMS__]
  
  def transform_to_host_BODY_from_NED(self, pos_host_NED: np.ndarray, ang_host_NED: float, vec_NED) -> np.ndarray:
    T_host_BODY_NED = np.array([
      [cos(ang_host_NED), -sin(ang_host_NED), pos_host_NED[0]],
      [sin(ang_host_NED), cos(ang_host_NED), pos_host_NED[1]],
      [0, 0, 1]
    ])
    return np.dot(np.linalg.inv(T_host_BODY_NED), np.concatenate((vec_NED, np.array([1]))))[0:Environment.__SPATIAL_DIMS__]

  def plot(self, axis, pos_host_NED: np.ndarray, ang_host_NED: float) -> None:
    pos_NED = self.transform_to_NED_from_host_BODY(pos_host_NED, ang_host_NED, self.self_pos_host_BODY)
    ang_NED = ang_host_NED + self.ang_host_BODY_self_BODY
    self.figs = []
    for ang in np.arange(-self.arc_angle/2, self.arc_angle/2 + self.angle_res, self.angle_res):
      self.figs.append(
        axis.plot(
          [pos_NED[0], pos_NED[0] + self.max_range*cos(ang + ang_NED)],
          [pos_NED[1], pos_NED[1] + self.max_range*sin(ang + ang_NED)],
          color="r", alpha=0.5)[0]
      )

  def update_plot(self, pos_host_NED: np.ndarray, ang_host_NED: float) -> None:
    pos_NED = self.transform_to_NED_from_host_BODY(pos_host_NED, ang_host_NED, self.self_pos_host_BODY)
    ang_NED = ang_host_NED + self.ang_host_BODY_self_BODY
    i = 0
    for ang in np.arange(-self.arc_angle/2, self.arc_angle/2 + self.angle_res, self.angle_res):
      self.figs[i].set_data(
        [pos_NED[0], pos_NED[0] + self.max_range*cos(ang + ang_NED)],
        [pos_NED[1], pos_NED[1] + self.max_range*sin(ang + ang_NED)]
      )
      i += 1

class ZeroRangeException(Exception):
  def __init__(self, msg="Zero range detected"):
    self.msg = msg
    super().__init__(msg)

class FullRangeException(Exception):
  def __init__(self, msg="Nothing detected within range"):
    self.msg = msg
    super().__init__(msg)