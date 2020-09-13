from typing import List, Tuple
from math import floor
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

class Environment():
  """
    Class representing the physical environment in which the CrazyFlies operate.
    Indexing parameter is passed as a tuple of floats representing the (x [m], y[m]) position
    of a cell in the environment.
    ...

    Attributes
    ----------
    x_res: float
      Number representing how many meters in the x-direction (horizontal) a cell in the occupancy grid span
    y_res: float
      Number representing how many meters in the y-direction (vertical) a cell in the occupancy grid span
    map: list[list[bool]]
      Two dimensional list containing the same information as occupancy_grid (passed as parameter to constructor), but
      keys are floats represetning the x- and y-position (in meters) of the cell.
    __SPATIAL_DIMS__: int
      number of spatial dimensions

    Methods
    ----------
    __init__(ocgit cupancy_grid, map_resolution): None
      Takes as argument an n-by-m matrix of booleans and a tuple of resolutions and computes an n-by-m dictionary 'map' of booleans

    __getitem__(coords): bool
      called by running environment[x, y]

      Takes as argument a tuple of coords (x[m], y[m]) and returns a boolean value
      signifying wether or not the cell at point (x, y) is occupied or not. If
      (x, y) is outside bounds of map it returns false
    
    __setitem__(coords, bool)
      called by running environment[x, y] = b

      Takes as argument a tuple of coords (x[m], y[m]) and sets the value
      of the cell occupying said coordinates to the value of the boolean b.
      Throws IndexError of coords are outside of bounds

    get_size(): :Tuple(float, float)
      returns size of environment in meters

    plot(axis): None
      Takes as argument a matplotlib.Axes object and plots the map on said axis
  """

  __SPATIAL_DIMS__ = 2

  def __init__(self, objects: List) -> None:
    self.objects = objects

  def add_object(self, object):
    self.objects.append(object)

  def get_objects(self):
    return self.objects.copy()

  def plot(self, axis: plt.Axes) -> None:
    for obj in self.objects:
      points = get_2d_points(obj)
      axis.plot(points[0,:], points[1,:],color="red")



def get_2d_points(object):
  if object["shape"] == "rectangle":
    points = object['points']
    points_unique = np.unique(points[0:2,:],axis=1)
    return points_unique
  else:
    return None


def main():
  points1 = np.array([[0,0,10,10],[0,0,0,0],[0,3,0,3]])
  points2 = np.array([[0,0,0,0],[0,0,10,10],[0,3,0,3]])
  points3 = np.array([[10,10,10,10],[0,0,10,10],[0,3,0,3]])
  points4 = np.array([[0,0,10,10],[10,10,10,10],[0,3,0,3]])

  obj1 = {"shape": "rectangle","points":points1}
  obj2 = {"shape": "rectangle", "points": points2}
  obj3 = {"shape": "rectangle", "points": points3}
  obj4 = {"shape": "rectangle", "points": points4}

  objects = [obj1,obj2,obj3,obj4]

  env = Environment(objects)

  fig,axis = plt.subplots()

  env.plot(axis)
  plt.show()
