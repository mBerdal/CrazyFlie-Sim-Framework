from abc import ABC, abstractmethod

class Drone(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def read_sensors(self):
        pass

    @abstractmethod
    def update_command(self):
        pass
    
    @abstractmethod
    def get_specs_dict(self):
      return {"class": type(self).__name__}