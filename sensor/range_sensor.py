from sensor.sensor import Sensor
from logger.log_entry import LogEntry
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
    
    """

    def __init__(self, sensor_pos_bdy: np.ndarray, sensor_attitude_bdy: np.ndarray, noise_generator = lambda: 0, **kwargs) -> None:
        super().__init__(noise_generator)
        self.max_range = kwargs.get("max_range", 4)
        self.min_range = kwargs.get("min_range", 0.04)
        self.arc_angle = kwargs.get("arc_angle", np.deg2rad(27))
        self.num_rays = kwargs.get("num_rays", 11)
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
          module=getmodule(self).__name__,
          cls = type(self).__name__,
          max_range = self.max_range,
          arc_angle = self.arc_angle,
          sensor_pos_bdy = self.sensor_pos_bdy,
          sensor_attitude_bdy = self.sensor_attitude_bdy
        )

    def get_reading(self) -> float:
      return self.measurement + self.noise_generator()

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
    def init_plot(axis, state, sensor_pos_bdy, sensor_attitude_bdy, max_range, arc_angle, measurement):
        w =  Wedge(
          (state[0:3] + rot_matrix_zyx(state.item(3), state.item(4), state.item(5))@sensor_pos_bdy)[0:2].reshape(2, ),
          measurement if measurement < max_range else max_range,
          np.rad2deg(state.item(5) + sensor_attitude_bdy.item(2) - arc_angle/2),
          np.rad2deg(state.item(5) + sensor_attitude_bdy.item(2) + arc_angle/2),
          color="red" if measurement < max_range else "blue",
          alpha=0.2
        )
        axis.add_patch(w)
        return w
        
    @staticmethod
    def update_plot(fig, state, sensor_pos_bdy, sensor_attitude_bdy, max_range, arc_angle, measurement):
        fig.set_center((state[0:3] + rot_matrix_zyx(state.item(3), state.item(4), state.item(5))@sensor_pos_bdy)[0:2].reshape(2, ))
        fig.set_radius(measurement if measurement < max_range else max_range)
        fig.set_theta1(np.rad2deg(state.item(5) + sensor_attitude_bdy.item(2) - arc_angle/2))
        fig.set_theta2(np.rad2deg(state.item(5) + sensor_attitude_bdy.item(2) + arc_angle/2))
        fig.set_color("red" if measurement < max_range else "blue")
        return fig






