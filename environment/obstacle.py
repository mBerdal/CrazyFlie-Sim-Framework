
import numpy as np

class Obstacle():
  def __init__(self, shape, points):
    self.shape = shape
    self.points = points

  def plot(self, axis):
    points = self.__get_2d_points()
    axis.plot(points[0,:], points[1,:], color="black")

  def __get_2d_points(self):
    if self.shape == "rectangle":
      return np.unique(self.points[0:2,:], axis=1)
    if self.shape == "triangle":
      return self.points
    else:
      return None