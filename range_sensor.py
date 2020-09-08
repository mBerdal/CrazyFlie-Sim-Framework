from sensor import Sensor
from typing import Tuple
import numpy as np
from math import cos, sin, atan
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from utils import rot_matrix_zyx

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

    def __init__(self, max_range: float, range_res: float, arc_angle: float, angle_res: float,
                 sensor_pos_bdy: np.ndarray, sensor_attitude_body: np.ndarray) -> None:
        super().__init__()
        self.max_range = max_range
        self.range_res = range_res
        self.arc_angle = arc_angle
        self.angle_res = angle_res
        self.num_beams = 3
        self.sensor_pos_bdy = sensor_pos_bdy
        self.sensor_attitude_bdy = sensor_attitude_body

    def get_reading(self, environment: Environment, state_host: np.ndarray) -> np.ndarray:
        # assert pos_host_NED.shape == (Environment.__SPATIAL_DIMS__, ), f"host position has shape {pos_host_NED.shape}, should be {(Environment.__SPATIAL_DIMS__, )}"
        rot_sensor_to_body = rot_matrix_zyx(self.sensor_attitude_bdy.item(0), self.sensor_attitude_bdy.item(1),
                                            self.sensor_attitude_bdy.item(2))
        rot_body_to_ned = rot_matrix_zyx(state_host.item(3), state_host.item(4), state_host.item(5))
        rot_sensor_to_ned = np.matmul(rot_body_to_ned, rot_sensor_to_body)

        pos_sensor_ned = state_host[0:3] + np.matmul(rot_body_to_ned, self.sensor_pos_bdy).reshape(3,1)
        beams = []
        for ang in np.arange(-self.arc_angle / 2, self.arc_angle / 2 + self.angle_res, self.angle_res):
            rot_beam_to_sensor = rot_matrix_zyx(0,0,ang)
            rot_beam_to_ned = np.matmul(rot_sensor_to_ned, rot_beam_to_sensor)
            beam = self.trace_beam(environment,pos_sensor_ned,rot_beam_to_ned)
            beams.append(beam)
        return beams

    def trace_beam(self, environment: Environment, pos_sensor_ned: np.ndarray, rot_beam_to_ned: np.ndarray):
        coarse_range = np.array([[min(environment.x_res, environment.y_res)], [0], [0]])
        converged = False
        coarse_beam = np.zeros((3, 1))
        while not converged and coarse_beam[0] < self.max_range:
            coarse_beam = coarse_beam + coarse_range
            pos_beam_ned = np.matmul(rot_beam_to_ned, coarse_beam).reshape(3,1) + pos_sensor_ned
            if environment[(pos_beam_ned.item(0), pos_beam_ned.item(1))]:
                converged = True

        if converged:
            beam_lower = coarse_beam - coarse_range
            beam_upper = coarse_beam
            while (beam_upper - beam_lower)[0] > self.range_res:
                midpoint_beam = (beam_upper - beam_lower) / 2 + beam_lower
                pos_midpoint_ned = np.matmul(rot_beam_to_ned, midpoint_beam).reshape(3,1) + pos_sensor_ned
                if environment[(pos_midpoint_ned.item(0), pos_midpoint_ned.item(1))]:
                    beam_upper = midpoint_beam
                else:
                    beam_lower = midpoint_beam
            return pos_midpoint_ned
        else:
            return self.max_range

    def plot(self, axis, pos_host_NED: np.ndarray, ang_host_NED: float) -> None:
        pos_NED = self.transform_to_NED_from_host_BODY(pos_host_NED, ang_host_NED, self.self_pos_host_BODY)
        ang_NED = ang_host_NED + self.ang_host_BODY_self_BODY
        self.figs = []
        for ang in np.arange(-self.arc_angle / 2, self.arc_angle / 2 + self.angle_res, self.angle_res):
            self.figs.append(
                axis.plot(
                    [pos_NED[0], pos_NED[0] + self.max_range * cos(ang + ang_NED)],
                    [pos_NED[1], pos_NED[1] + self.max_range * sin(ang + ang_NED)],
                    color="r", alpha=0.5)[0]
            )

    def update_plot(self, pos_host_NED: np.ndarray, ang_host_NED: float) -> None:
        pos_NED = self.transform_to_NED_from_host_BODY(pos_host_NED, ang_host_NED, self.self_pos_host_BODY)
        ang_NED = ang_host_NED + self.ang_host_BODY_self_BODY
        i = 0
        for ang in np.arange(-self.arc_angle / 2, self.arc_angle / 2 + self.angle_res, self.angle_res):
            self.figs[i].set_data(
                [pos_NED[0], pos_NED[0] + self.max_range * cos(ang + ang_NED)],
                [pos_NED[1], pos_NED[1] + self.max_range * sin(ang + ang_NED)]
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
