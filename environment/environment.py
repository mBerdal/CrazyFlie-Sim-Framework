from typing import List, Tuple
from math import floor
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from environment.obstacle import Obstacle


class Environment():
    __SPATIAL_DIMS__ = 2

    def __init__(self, obstacles=[]) -> None:
        self.obstacles = []
        for o in obstacles:
            self.add_obstacle(o)

    def add_obstacle(self, obstacle):
        if obstacle.shape == "rectangle":
            self.add_obstacle(Obstacle("triangle", obstacle.points[:, 0:3]))
            self.add_obstacle(Obstacle("triangle", obstacle.points[:, 1:4]))
        elif obstacle.shape == "triangle":
            self.obstacles.append(obstacle)

    def get_obstacles(self):
        return self.obstacles

    def plot(self, axis: plt.Axes) -> None:
        for obstacle in self.obstacles:
            obstacle.plot(axis)

    def to_JSONable(self):
        return {
            "obstacles": [
                {
                    "shape": obstacle.shape,
                    "points": obstacle.points.tolist()
                } for obstacle in self.obstacles
            ]
        }
