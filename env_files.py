from environment.environment import Environment
from environment.obstacle import Obstacle
import numpy as np
from utils.misc import create_straight_wall

"""
Storage file for environments used during simulations. 
"""

w = 0.1
h = 3

corner1 = [np.array([-15,-15,0]),
           np.array([-15,-15,0]),
           np.array([15,-15,0]),
           np.array([-15,15,0]),
           np.array([-15,-10,0]),
           np.array([-10,-11.5,0]),
           np.array([-10,-15,0]),
           np.array([-15,-3,0]),
           np.array([-10,-6,0]),
           np.array([-10-w,2,0]),
           np.array([-15,5,0]),
           #np.array([-5,-15,0]),
           np.array([-5,-10,0]),
           np.array([10,-15,0]),
           np.array([6,-10,0]),
           np.array([6,-5,0]),
           np.array([-2,5,0]),
           np.array([-2,5,0]),
           np.array([-2,0,0]),
           np.array([3,3,0]),
           np.array([-2,-7,0]),
           np.array([12,-5,0]),
           np.array([12,1,0]),
           np.array([-15,10,0]),
           np.array([-11,10,0]),
           np.array([-5,10,0]),
           np.array([-7,5,0]),
           np.array([0,15,0]),
           np.array([4,15,0]),
           np.array([4,8,0]),
           np.array([10,8,0]),
           np.array([6,12,0]),
           np.array([12,-8,0]),
           np.array([3,-7,0]),
           np.array([0,-15, 0])



           ]

corner2 = [np.array([-15+w,15,h]),
           np.array([15,-15+w,h]),
           np.array([15-w,15,h]),
           np.array([15,15-w,h]),
           np.array([-10,-10+w,h]),
           np.array([-10-w,-7.5,h]),
           np.array([-10-w,-13,h]),
           np.array([-10,-3+w,h]),
           np.array([-10-w,2,h]),
           np.array([-8,2+w,h]),
           np.array([-8.5,5+w,h]),
           #np.array([-5+w,-11,h]),
           np.array([6,-10+w,h]),
           np.array([10+w,-11,h]),
           np.array([6-w,5,h]),
           np.array([12,-5+w,h]),
           np.array([12, 5+w,h]),
           np.array([-2+w,-7,h]),
           np.array([3,0+w,h]),
           np.array([3+w,-7,h]),
           np.array([0,-7+w,h]),
           np.array([12-w,-0.5,h]),
           np.array([12-w,5,h]),
           np.array([-12.5,10+w,h]),
           np.array([-5,10+w,h]),
           np.array([-5-w,5,h]),
           np.array([-5,5+w,h]),
           np.array([0+w,8,h]),
           np.array([4+w,8,h]),
           np.array([8,8+w,h]),
           np.array([15,8+w,h]),
           np.array([15,12+w,h]),
           np.array([15,-8+w,h]),
           np.array([1.5,-7+w,h]),
           np.array([w,-12.5,h])

           ]

ENV_OFF = Environment()

for i in range(len(corner1)):
    obs = create_straight_wall(corner1[i],corner2[i])
    for o in obs:
        o = Obstacle("rectangle",o)
        ENV_OFF.add_obstacle(o)


corner1 = [np.array([-15,-15,0]),
           np.array([-15,-15,0]),
           np.array([15,-15,0]),
           np.array([-15,15,0]),

           np.array([-12,-12,0]),
           np.array([-12,-7,0]),
           np.array([-7,-7,0]),
           np.array([-7,-12,0]),

           np.array([-12,-4,0]),
           np.array([-12,1,0]),
           np.array([-7,1,0]),
           np.array([-7,-4,0]),

           np.array([7,-12,0]),
           np.array([7, 12,0]),
           np.array([12,12,0]),
           np.array([12,-12,0]),

           np.array([-4,-12,0]),
           np.array([-4, -7,0]),
           np.array([4, -7,0]),
           np.array([4,-12,0]),

           np.array([-12,4,0]),
           np.array([-12, 12,0]),
           np.array([-7, 12,0]),
           np.array([-7, 4,0]),

           np.array([-4,-1.5,0]),
           np.array([-4, 12,0]),
           np.array([0, 12,0]),
           np.array([0, -1.5,0]),


           ]

corner2 = [np.array([-15+w,15,h]),
           np.array([15,-15+w,h]),
           np.array([15-w,15,h]),
           np.array([15,15-w,h]),

           np.array([-12+w,-7,h]),
           np.array([-7,-7-w,h]),
           np.array([-7-w,-12,h]),
           np.array([-12,-12+w,h]),

           np.array([-12+w,1,h]),
           np.array([-7,1-w,h]),
           np.array([-7-w,-4,h]),
           np.array([-12,-4+w,h]),

           np.array([7+w,12,h]),
           np.array([12,12-w,h]),
           np.array([12-w,-12,h]),
           np.array([7,-12+w,h]),

           np.array([-4+w,-7,h]),
           np.array([4, -7-w,h]),
           np.array([4-w, -12,h]),
           np.array([-4,-12+w,h]),

           np.array([-12+w,12,h]),
           np.array([-7, 12-w,h]),
           np.array([-7+w, 4,h]),
           np.array([-12, 4+w,h]),

           np.array([-4+w,12,h]),
           np.array([0, 12-w,h]),
           np.array([0-w, -1.5,h]),
           np.array([-4, -1.5+w,h]),
           ]


ENV_LOOP = Environment()

for i in range(len(corner1)):
    obs = create_straight_wall(corner1[i],corner2[i])
    for o in obs:
        o = Obstacle("rectangle",o)
        ENV_LOOP.add_obstacle(o)
