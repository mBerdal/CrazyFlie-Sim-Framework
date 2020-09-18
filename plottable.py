from abc import ABC, abstractmethod

class Plottable(ABC):
  @staticmethod
  @abstractmethod
  def init_plot(axis, **kwargs):
    pass

  @staticmethod
  @abstractmethod
  def update_plot(fig, **kwargs):
    pass
