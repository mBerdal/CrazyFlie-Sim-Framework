from environment.environment import Environment
from environment.obstacle import Obstacle
from drone_swarm.drone.crazy_flie import CrazyFlieLidar
from simulator import Simulator
from controller import SwarmExplorationController

import numpy as np
from utils.misc import create_straight_wall

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
           np.array([3,-7,0])



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
           np.array([1.5,-7+w,h])

           ]

env = Environment()

for i in range(len(corner1)):
    obs = create_straight_wall(corner1[i],corner2[i])
    for o in obs:
        o = Obstacle("rectangle",o)
        env.add_obstacle(o)

x = -13
y = 13
z = 0.1
yaw = 0

drone1 = CrazyFlieLidar(0,np.array([x,y,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)
drone2 = CrazyFlieLidar(1,np.array([x+1,y,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)
drone3 = CrazyFlieLidar(2,np.array([x,y+1,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)

initial_pose_1 = np.array([x,y,yaw],dtype=float).reshape(3,1)
initial_pose_2 = np.array([x+1,y,yaw],dtype=float).reshape(3,1)
initial_pose_3 = np.array([x,y+1,yaw],dtype=float).reshape(3,1)

rays = drone1.sensors[0].ray_vectors

drones = [drone1, drone2, drone3]
initial_poses = [initial_pose_1, initial_pose_2, initial_pose_3]

if __name__ == "__main__":
    c = SwarmExplorationController(drones,initial_poses, rays,
                                   visualize=False, num_particles=10, assignment_method="optimized")
    s = Simulator(environment=env, drones=drones, controller=c, com_delay=0.1, log_to_file="logs/test.json",
                  env_to_file="logs/env_test_4.json", con_to_file="logs/controller.json",slam_to_file="logs/slam.json",
                  shared_map_to_file="logs/shared_map.json")
    s.simulate(0.05, 200)
    #s.visualize()
