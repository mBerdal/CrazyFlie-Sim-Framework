from abc import ABC, abstractmethod
from log_entry import Loggable


class Sensor(Loggable, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_reading(self, environment):
        pass