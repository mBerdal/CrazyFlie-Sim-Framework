from sensor import Sensor
from typing import Tuple
import numpy as np
from math import cos, sin, atan
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from utils.rotation_utils import rot_matrix_zyx
from utils.raytracing import intersect_rectangle
import multiprocessing as mp
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
    get_reading(environment, state_host): np.ndarray/None
      Takes as argument an environment object, state of host represented in the NED coordinate system. If there are obstacles
      in the environment that are closer to the sensor than max_range and within field of view it returns
      a vector from the origin of host's coordinate to the nearest obstacle. If there are no obstacles closer
      than max_range and within field of view it returns None
    """

    def __init__(self, sensor_pos_bdy: np.ndarray, sensor_attitude_body: np.ndarray,**kwargs) -> None:
        super().__init__()
        self.max_range = kwargs.get("max_range",4)
        self.min_range = kwargs.get("min_range",0.04)
        self.arc_angle = kwargs.get("arc_angle",np.deg2rad(27))
        self.num_rays = kwargs.get("num_rays",11)
        self.sensor_pos_bdy = sensor_pos_bdy
        self.sensor_attitude_bdy = sensor_attitude_body
        self.ray_vectors, self.ray_orgins = self.calculate_ray_vectors()
        self.max_range_vector = np.ones(self.ray_vectors.shape[1])*self.max_range
        self.min_range_vector = np.ones(self.ray_vectors.shape[1])*self.min_range

    def get_reading(self, objects, state_host: np.ndarray, return_all_beams = False) -> np.ndarray:
        pass

    def get_ray_vectors(self):
        return self.ray_vectors, self.ray_orgins, self.max_range_vector, self.min_range_vector

    def get_specs_dict(self):
        return {**super().get_specs_dict(), **{
          "max_range": self.max_range,
          "arc_angle": self.arc_angle,
          "sensor_pos_bdy": self.sensor_pos_bdy.tolist(),
          "sensor_attitude_bdy": self.sensor_attitude_bdy.tolist()
        }}

    def calculate_ray_vectors(self):
        rot_sensor_to_body = rot_matrix_zyx(self.sensor_attitude_bdy.item(0), self.sensor_attitude_bdy.item(1),
                                            self.sensor_attitude_bdy.item(2))
        vectors = np.zeros((3,self.num_rays))
        for ind, ang in enumerate(np.linspace(-self.arc_angle/2,self.arc_angle/2,self.num_rays)):
            #Get rotation matrix from beam frame to ned frame
            rot_ray_to_sensor = rot_matrix_zyx(0,0,ang)
            rot_ray_to_body = np.matmul(rot_sensor_to_body, rot_ray_to_sensor)
            vectors[:,ind] = rot_ray_to_body @ np.array([1,0,0])
        orgins = np.array([self.sensor_pos_bdy]*self.num_rays,np.float).squeeze().transpose()
        return vectors, orgins

    def plot(self, axis, environment, state_host: np.ndarray) -> None:
        pass

    def update_plot(self, environment, state_host: np.ndarray) -> None:
        pass






