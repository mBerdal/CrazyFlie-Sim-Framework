from abc import ABC, abstractmethod

class Plottable(ABC):
  @staticmethod
  @abstractmethod
  def init_plot(axis, state, **kwargs):
    pass

  @staticmethod
  @abstractmethod
  def update_plot(fig, state, **kwargs):
    pass
