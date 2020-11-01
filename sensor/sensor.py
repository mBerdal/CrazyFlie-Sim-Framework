from logger.loggable import Loggable
from logger.log_entry import LogEntry

from copy import deepcopy
from abc import ABC, abstractmethod


class Sensor(Loggable, ABC):
    def __init__(self, noise_generator=lambda: 0):
        self.measurement = None
        self.noise_generator = noise_generator
        super().__init__()

    @abstractmethod
    def get_reading(self, environment):
        pass

    @abstractmethod
    def add_measurement(self, measurements):
        pass

    def generate_time_entry(self, measurement):
        return LogEntry(
            measurement=measurement
        )

    def get_time_entry(self):
        return LogEntry(
            measurement=deepcopy(self.measurement)
        )
