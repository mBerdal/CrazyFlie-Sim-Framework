from sensor.sensor import Sensor
import numpy as np
from logger.log_entry import LogEntry
from inspect import getmodule

from utils.rotation_utils import rot_matrix_2d

class OdometrySensor(Sensor):

    def __init__(self, noise_generator=lambda: noise_odometry(),**kwargs):
        super().__init__(noise_generator)
        self.measurement = np.zeros([3,1])

    def get_reading(self):
        return {"type": type(self), "reading": self.measurement.copy()}

    def add_measurement(self, state_dot, time_step):
        noise = noise_odometry()
        noisy_state_dot = (state_dot[[0,1,5]] + noise)
        self.measurement += time_step * noisy_state_dot


    def get_info_entry(self):
        return LogEntry(
            module=getmodule(self).__name__,
            cls=type(self).__name__
        )

def noise_odometry():
    n = np.zeros([3,1])
    n[0] = np.random.normal(scale=0.2)
    n[1] = np.random.normal(scale=0.2)
    n[2] = np.random.normal(scale=0.1)
    return n