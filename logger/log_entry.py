from enum import Enum
import numpy as np


class LogEntry():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      self.__dict__[k] = v

  def to_JSONable(self) -> dict:
    d = self.__dict__
    for k, v in d.items():
      if isinstance(v, LogEntry):
        d[k] = v.to_JSONable()
      elif isinstance(v, list):
        d[k] = [item.to_JSONable() if isinstance(item, LogEntry) else item for item in v]
      elif isinstance(v, np.ndarray):
        d[k] = v.tolist()
    return d