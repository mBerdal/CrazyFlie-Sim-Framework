from typing import List, Tuple
from math import floor
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Environment():
  """
    Class representing the physical environment in which the CrazyFlies operate
    ...

    Attributes
    ----------
    x_res: float
      Number representing how many meters in the x-direction (horizontal) a cell in the occupancy grid span
    y_res: float
      Number representing how many meters in the y-direction (vertical) a cell in the occupancy grid span
    map: dict{dict{bool}}
      Two dimensional dictionary containing the same information as occupancy_grid (passed as parameter to constructor), but
      keys are floats represetning the x- and y-position (in meters) of the cell.
    __SPATIAL_DIMS__: int
      number of spatial dimensions

    Methods
    ----------
    __init__(occupancy_grid, map_resolution): None
      Takes as argument an n-by-m matrix of booleans and a tuple of resolutions and computes an n-by-m dictionary 'map' of booleans

    is_cell_at_coords_occupied(x, y): boolean
      Takes as argument a point (x [m], y [m]) and returns wether or not the cell spanning said point contains an obstacle

    plot(axis): None
      Takes as argument a matplotlib.Axes object and plots the map on said axis
  """

  __SPATIAL_DIMS__ = 2

  def __init__(self, occupancy_grid: List[List[bool]], map_resolution: Tuple = (0.08, 0.08)) -> None:
    self.x_res, self.y_res = map_resolution

    self.map = dict.fromkeys([self.y_res*y for y in range(len(occupancy_grid))])
    for y in range(len(occupancy_grid)):
      self.map[y*self.y_res] = dict.fromkeys([self.x_res*x for x in range(len(occupancy_grid[y]))])
      for x in range(len(occupancy_grid[y])):
        self.map[y*self.y_res][x*self.x_res] = occupancy_grid[y][x]


  def is_cell_at_coords_occupied(self, x: float, y: float) -> bool:
    try:
      return self.map[floor(y/self.y_res)*self.y_res][floor(x/self.x_res)*self.x_res]
    except KeyError:
      return True

  def plot(self, axis: plt.Axes) -> None:
    for y, x_dict in self.map.items():
      for x, is_obstacle in x_dict.items():
        if is_obstacle:
          axis.add_patch(patches.Rectangle((x, y), self.x_res, self.y_res))