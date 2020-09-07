from abc import ABS, abstractmethod

class Logger(ABC):
    """
  Class representing a logger
  ...

  Attributes
  ----------
  file: TODO: what type of file? .txt, .csv?
  
  Methods
  ----------
  read_log()
   Read from the log

  write_to_log()
    Write to the log
"""

    def __init__(self):
        pass

    @abstractmethod
    def read_log(self):
        pass

    @abstractmethod
    def write_to_log(self)
        pass