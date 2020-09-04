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
    __init__(occupancy_grid, map_resolution): None
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

  def __init__(self, occupancy_grid: List[List[bool]], map_resolution: Tuple = (0.08, 0.08)) -> None:
    self.x_res, self.y_res = map_resolution
    self.map = list(reversed(occupancy_grid))

  def __getitem__(self, coords: Tuple) -> bool:
    x, y = coords
    assert(isinstance(x, float) or isinstance(x, int))\
      and (isinstance(y, float) or isinstance(y, int))\
      ,"slicing not supported. Use get_section to obtain section of environment"
    try:
      return self.map[floor(y/self.y_res)][floor(x/self.x_res)]
    except IndexError as e:
      print(f"Warning. Tried to look up coordinates ({x}m, {y}m) which is outside map. Returning True")
      return True

  def __setitem__(self, coords: Tuple, value) -> None:
    x, y = coords
    self.map[floor(y/self.y_res)][floor(x/self.x_res)] = value

  def __delitem__(self, coords: Tuple) -> None:
    x, y = coords
    self.map[floor(y/self.y_res)][floor(x/self.x_res)] = False

  def get_size(self):
    try:
      return len(self.map[0])*self.x_res, len(self.map)*self.y_res
    except IndexError:
      return 0, 0

  def plot(self, axis: plt.Axes) -> None:
    for y in range(len(self.map)):
      for x in range(len(self.map[y])):
        if self.map[y][x]:
          axis.add_patch(patches.Rectangle((x*self.x_res, y*self.y_res), self.x_res, self.y_res))