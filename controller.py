from abc import ABC, abstractmethod

class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data):
        pass