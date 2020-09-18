from sensor.sensor import Sensor
from logger.log_entry import LogEntry, EntryType
from utils.rotation_utils import rot_matrix_zyx
from plottable import Plottable

from inspect import getmodule
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from copy import deepcopy

class RangeSensor(Sensor, Plottable):
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

    def __init__(self, sensor_pos_bdy: np.ndarray, sensor_attitude_bdy: np.ndarray,**kwargs) -> None:
        super().__init__()
        self.max_range = kwargs.get("max_range",4)
        self.min_range = kwargs.get("min_range",0.04)
        self.arc_angle = kwargs.get("arc_angle",np.deg2rad(27))
        self.num_rays = kwargs.get("num_rays",11)
        self.sensor_pos_bdy = sensor_pos_bdy
        self.sensor_attitude_bdy = sensor_attitude_bdy
        assert (not self.sensor_attitude_bdy is None) and (not self.sensor_pos_bdy is None),\
          "failed initing range_sensor. sensor_attitude_bdy and sensor_pos_bdy must be specified"
        self.ray_vectors, self.ray_orgins = self.__calculate_ray_vectors()
        self.max_range_vector = np.ones(self.ray_vectors.shape[1])*self.max_range
        self.min_range_vector = np.ones(self.ray_vectors.shape[1])*self.min_range
        self.measurement = float("inf")

    def get_info_entry(self):
        return LogEntry(
          EntryType.INFO,
          module=getmodule(self).__name__,
          cls = type(self).__name__,
          max_range = self.max_range,
          arc_angle = self.arc_angle,
          sensor_pos_bdy = self.sensor_pos_bdy,
          sensor_attitude_bdy = self.sensor_attitude_bdy
        )

    def get_reading(self, objects, state_host: np.ndarray, return_all_beams = False) -> np.ndarray:
        pass

    def get_ray_vectors(self):
        return self.ray_vectors, self.ray_orgins, self.max_range_vector, self.min_range_vector


    def __calculate_ray_vectors(self):
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

    @staticmethod
    def init_plot(axis, host_state, host_rel_pos, host_rel_attitude, max_range, arc_angle, measured_range):
        w =  Wedge(
          (host_state[0:3] + rot_matrix_zyx(host_state.item(3), host_state.item(4), host_state.item(5))@host_rel_pos)[0:2].reshape(2, ),
          measured_range if measured_range < max_range else max_range,
          np.rad2deg(host_state.item(5) + host_rel_attitude.item(2) - arc_angle/2),
          np.rad2deg(host_state.item(5) + host_rel_attitude.item(2) + arc_angle/2),
          color="red" if measured_range < max_range else "blue",
          alpha=0.3
        )
        axis.add_patch(w)
        return w
        
    @staticmethod
    def update_plot(fig, host_state, host_rel_pos, host_rel_attitude, max_range, arc_angle, measured_range):
        fig.set_center((host_state[0:3] + rot_matrix_zyx(host_state.item(3), host_state.item(4), host_state.item(5))@host_rel_pos)[0:2].reshape(2, ))
        fig.set_radius(measured_range if measured_range < max_range else max_range)
        fig.set_theta1(np.rad2deg(host_state.item(5) + host_rel_attitude.item(2) - arc_angle/2))
        fig.set_theta2(np.rad2deg(host_state.item(5) + host_rel_attitude.item(2) + arc_angle/2))
        fig.set_color("red" if measured_range < max_range else "blue")
        return fig






