from abc import ABC, abstractmethod
from logger.log_entry import LogEntry
from logger.loggable import Loggable
from plottable import Plottable
from communication import CommunicationNode
import numpy as np

from copy import deepcopy
from inspect import getmodule


class Drone(CommunicationNode, Loggable, Plottable, ABC):

    def __init__(self, d_id, state, sensors, state_noise_generator=lambda: np.zeros(self.state.shape)):
        self.id = d_id
        self.state = state
        self.state_dot = np.zeros([6, 1])
        self.sensors = sensors
        self.state_noise_generator = state_noise_generator
        super().__init__()

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def read_sensors(self):
        pass

    def get_state_reading(self):
        return np.concatenate([deepcopy(self.state) + self.state_noise_generator(),
                               deepcopy(self.state_dot) + self.state_noise_generator()], axis=0)

    def get_info_entry(self):
        return LogEntry(
            module=getmodule(self).__name__,
            cls=type(self).__name__,
            id=self.id,
            sensors=[
                s.get_info_entry() for s in self.sensors
            ]
        )

    def get_time_entry(self):
        return LogEntry(
            id=self.id,
            state=deepcopy(self.state),
            measurements=[
                s.get_time_entry() for s in self.sensors
            ]
        )

    def generate_time_entry(self, state, measurements):
        return LogEntry(
            id=self.id,
            state=state,
            measurements=[
                self.sensors[i].generate_time_entry(measurements[i]) for i in range(len(self.sensors))
            ]
        )
