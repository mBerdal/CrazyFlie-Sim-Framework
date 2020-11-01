from abc import ABC, abstractmethod


class Loggable(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_info_entry(self):
        pass

    @abstractmethod
    def generate_time_entry(self):
        pass

    @abstractmethod
    def get_time_entry(self):
        pass
