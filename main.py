from drone_swarm.drone.crazy_flie import CrazyFlieLidar
from simulator import Simulator
from controller import SwarmExplorationController

import numpy as np
from env_files import ENV_LOOP, ENV_OFF

env = ENV_LOOP #Either ENV_LOOP (hallway env with many loops) or ENV_OFF (office env with small rooms)

num_particles = 10 #Number of particles used in the particle filter. More than 15 is very slow.
num_drones = 1 #Number of drones. 1-4. More can be added, but require modification of code under (similar to the existing).

time = 180 #Number of seconds for the simulation
time_step = 0.05 #Timestep used in the simulation.

assingment_method = "optimized" # "optimized", "greedy" or "closest". "optimized" require that Gurobi for python is installed
visualize = True

#Set the file paths where you want to store the log. Set None if not needed.
log_slam = "logs/slam50.json"
log_shared_map = "logs/shared_map50.json"
log_controller = "logs/controller50.json"
env_file = "logs/env_test.json"
log_drones = "logs/test7.json"

#Initial positon for the drones. If multiple drones, the drones are initialized one meter from the given in a square
x = 0
y = -5
z = 0.1
yaw = 0

drone1 = CrazyFlieLidar(0,np.array([x,y,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)
drone2 = CrazyFlieLidar(1,np.array([x+1,y,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)
drone3 = CrazyFlieLidar(2,np.array([x,y+1,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)
drone4 = CrazyFlieLidar(3,np.array([x+1,y+1,z,0,0,yaw]).reshape(6,1),num_beams_sensor=360,max_range_sensor=10)

initial_pose_1 = np.array([x,y,yaw],dtype=float).reshape(3,1)
initial_pose_2 = np.array([x+1,y,yaw],dtype=float).reshape(3,1)
initial_pose_3 = np.array([x,y+1,yaw],dtype=float).reshape(3,1)
initial_pose_4 = np.array([x+1,y+1,yaw],dtype=float).reshape(3,1)

rays = drone1.sensors[0].ray_vectors

drones = [drone1, drone2, drone3, drone4]
initial_poses = [initial_pose_1,initial_pose_2, initial_pose_3, initial_pose_4]

drones = drones[0:num_drones]
initial_poses = initial_poses[0:num_drones]

if __name__ == "__main__":
    c = SwarmExplorationController(drones,initial_poses, rays,
                                   visualize=visualize, num_particles=num_particles, assignment_method=assingment_method)
    s = Simulator(environment=env, drones=drones, controller=c, com_delay=0.1, log_to_file=log_drones,
                  env_to_file=env_file, con_to_file=log_controller, slam_to_file=log_slam,
                  shared_map_to_file=log_shared_map)
    s.simulate(time_step, time)
    #s.visualize()
