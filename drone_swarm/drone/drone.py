from abc import ABC, abstractmethod
from logger.log_entry import LogEntry
from logger.loggable import Loggable
from plottable import Plottable
from communication import CommunicationNode
from numpy import zeros

from copy import deepcopy
from inspect import getmodule

class Drone(CommunicationNode, Loggable, Plottable, ABC):

    def __init__(self, d_id, eom, sensors, state_noise_generator = lambda: zeros(self.eom.get_state().shape)):
      self.id = d_id
      self.sensors = sensors
      self.state_noise_generator = state_noise_generator
      self.eom = eom
      super().__init__()

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def read_sensors(self):
        pass

    def get_state_reading(self):
      return deepcopy(self.eom.get_state()) + self.state_noise_generator()
    
    def get_info_entry(self):
      return LogEntry(
        module=getmodule(self).__name__,
        cls = type(self).__name__,
        id = self.id,
        sensors = [
          s.get_info_entry() for s in self.sensors
        ]
      )
  
    def get_time_entry(self):
      return LogEntry(
        id = self.id,
        state = self.eom.get_state(),
        measurements = [
          s.get_time_entry() for s in self.sensors
        ]
      )
    
    def generate_time_entry(self, state, measurements):
      return LogEntry(
        id = self.id,
        state = state,
        measurements = [
          self.sensors[i].generate_time_entry(measurements[i]) for i in range(len(self.sensors))
        ]
      )