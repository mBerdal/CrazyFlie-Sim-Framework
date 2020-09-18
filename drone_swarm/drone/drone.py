from abc import ABC, abstractmethod
from logger.log_entry import LogEntry, EntryType
from logger.loggable import Loggable
from copy import deepcopy
from inspect import getmodule

class Drone(Loggable, ABC):

    def __init__(self, d_id, state, sensors):
      self.id = d_id
      self.state = state
      self.sensors = sensors

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def read_sensors(self):
        pass

    @abstractmethod
    def update_command(self):
        pass
    
    def get_info_entry(self):
      return LogEntry(
        EntryType.INFO,
        module=getmodule(self).__name__,
        cls = type(self).__name__,
        id = self.id,
        sensors = [
          s.get_info_entry() for s in self.sensors
        ]
      )
  
    def get_time_entry(self):
      return LogEntry(
        EntryType.TIME,
        id = self.id,
        state = deepcopy(self.state),
        measurements = [
          s.get_time_entry() for s in self.sensors
        ]
      )
    
    def generate_time_entry(self, state, measurements):
      return LogEntry(
        EntryType.TIME,
        id = self.id,
        state = state,
        measurements = [
          self.sensors[i].generate_time_entry(measurements[i]) for i in range(len(self.sensors))
        ]
      )