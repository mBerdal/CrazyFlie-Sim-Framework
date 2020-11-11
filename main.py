from environment.environment import Environment
from environment.obstacle import Obstacle
from drone_swarm.drone.crazy_flie import CrazyFlieLidar
from simulator import Simulator
from controller import SwarmExplorationController

import numpy as np
from utils.misc import create_straight_wall

w = 0.1
h = 3
corner1 = [np.array([0,0,0]),
           np.array([0,0,0]),
           np.array([0,20,0]),
           np.array([20,0,0]),
           np.array([0,10,0]),
           np.array([0,16,0]),
           np.array([4,10,0]),
           np.array([8,16,0]),
           np.array([17,16,0]),
           np.array([14,16,0]),
           np.array([14,8,0]),
           np.array([10,8,0]),
           np.array([10,6,0]),
           np.array([10,0,0])
           ]

corner2 = [np.array([w,20,h]),
           np.array([20,w,h]),
           np.array([20,20-w,h]),
           np.array([20-w,20,h]),
           np.array([2,10+w,h]),
           np.array([6,16+w,h]),
           np.array([10,10+w,h]),
           np.array([15,16+w,h]),
           np.array([20,16+w,h]),
           np.array([14+w,14,h]),
           np.array([14+w,12,h]),
           np.array([14,8+w,h]),
           np.array([10+w,16,h]),
           np.array([10+w,4,h])
           ]

env = Environment()

for i in range(len(corner1)):
    obs = create_straight_wall(corner1[i],corner2[i])
    for o in obs:
        o = Obstacle("rectangle",o)
        env.add_obstacle(o)

x = 2
y = 2
z = 0.1
yaw = 0

drone1 = CrazyFlieLidar(0,np.array([x,y,z,0,0,yaw]).reshape(6,1),num_beams_sensor=180,max_range_sensor=10)
drone2 = CrazyFlieLidar(1,np.array([x+1,y,z,0,0,yaw]).reshape(6,1),num_beams_sensor=180,max_range_sensor=10)
drone3 = CrazyFlieLidar(2,np.array([x,y+1,z,0,0,yaw]).reshape(6,1),num_beams_sensor=180,max_range_sensor=10)

initial_pose_1 = np.array([x,y,yaw]).reshape(3,1)
initial_pose_2 = np.array([x+1,y,yaw]).reshape(3,1)
initial_pose_3 = np.array([x,y+1,yaw]).reshape(3,1)



if __name__ == "__main__":
    c = SwarmExplorationController([drone1, drone2, drone3],[initial_pose_1, initial_pose_2, initial_pose_3],visualize=True)
    s = Simulator(environment=env, drones=[drone1, drone2, drone3], controller=c, com_delay=0.1, log_to_file="logs/test_lidar_a.json",
                  env_to_file="logs/env_test_1.json", con_to_file="logs/controller.json",slam_to_file="logs/slam.json",
                  shared_map_to_file="logs/shared_map.json")
    s.simulate(0.05, 30)
    s.visualize()
