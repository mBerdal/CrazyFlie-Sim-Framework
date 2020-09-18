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

  def __init__(self, objects = []) -> None:
    self.objects = []
    for o in objects:
      self.add_object(o)

  def add_object(self, obj):
    if obj["shape"] == "rectangle":
      self.add_rectangle(obj)

  def add_rectangle(self, obj):
    assert obj["shape"] == "rectangle"
    self.objects.append({"shape": "triangle","points":obj["points"][:,0:3]})
    self.objects.append({"shape": "triangle","points": obj["points"][:, 1:4]})

  def get_objects(self):
    return self.objects.copy()

  def plot(self, axis: plt.Axes) -> None:
    for obj in self.objects:
      points = get_2d_points(obj)
      axis.plot(points[0,:], points[1,:],color="red")

  def to_JSONable(self):
    return {
          "objects": [
            {
              "shape": obj["shape"],
              "points": obj["points"].tolist()
            } for obj in self.objects
          ]
        }
def get_2d_points(object):
  if object["shape"] == "rectangle":
    points = object['points']
    points_unique = np.unique(points[0:2,:],axis=1)
    return points_unique
  elif object["shape"] == "triangle":
    return object["points"]
  else:
    return None
