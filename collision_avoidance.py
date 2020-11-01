import numpy as np
from utils.rotation_utils import ssa
import matplotlib.pyplot as plt

class PredicativeCollisionAvoidance:

    def __init__(self):
        self.weight_avoidance = 10
        self.weight_navigation = 20
        self.weight_control = 1

        self.weight_velocity_command = 3
        self.weight_heading_command = 3

        self.heading = np.pi
        self.heading_num = 11
        self.heading_commands = np.linspace(-self.heading,self.heading,self.heading_num)

        self.velocity_commands = [0.5, 1]

        self.time_step_simulation = 0.1
        self.time_horizon = 3

        self.heading_delta_max = np.deg2rad(180)
        self.vel_delta_max = 2

        self.eps = 1e-6
        self.delta_pos = 0.5

    def collision_avoidance(self, pose, velocity, distance_grid, waypoint):
        scores = np.zeros([len(self.velocity_commands),len(self.heading_commands)])

        for v in range(len(self.velocity_commands)):
            for h in range(len(self.heading_commands)):
                trajectory = self.simulate_trajectories(pose.copy(), velocity, self.velocity_commands[v], self.heading_commands[h], waypoint)
                score_nav, score_avoid = self.score_trajectories(trajectory,distance_grid,waypoint)
                score_control = abs(1-self.velocity_commands[v])*self.weight_velocity_command + abs(self.heading_commands[h])*self.weight_heading_command
                scores[v, h] = score_nav*self.weight_navigation + score_avoid*self.weight_avoidance + score_control*self.weight_control

        ind = np.unravel_index(np.argmin(scores, axis=None), scores.shape)

        heading_command = self.heading_commands[ind[1]]
        velocity_command = self.velocity_commands[ind[0]]

        return heading_command, velocity_command

    def score_trajectories(self, trajectory, distance_grid, waypoint):
        navigation_score = 0
        avoidance_score = 0

        for t in trajectory:
            err_x = waypoint[0] - t[0]
            err_y = waypoint[1] - t[1]
            heading_waypoint = np.arctan2(err_y, err_x) % (2 * np.pi)
            navigation_score += abs(ssa(heading_waypoint,t[2]))
            x_cell = np.int(np.floor((t[0] - distance_grid["lower_x"])/distance_grid["res"]))
            y_cell = np.int(np.floor((t[1] - distance_grid["lower_y"]) / distance_grid["res"]))
            try:
                distance = distance_grid["grid"][x_cell,y_cell]
            except:
                distance = 2
            avoidance_score += 1/(distance+self.eps)**2

        return navigation_score, avoidance_score

    def simulate_trajectories(self, pose, velocity, velocity_command, heading_command, waypoint):
        current_velocity = velocity
        velocity_d = velocity_command

        current_heading = pose[2]
        heading_d = heading_command + current_heading

        current_pose = pose
        trajectory = list()

        trajectory.append(current_pose.copy())
        t = 0

        end_flag = False
        while t < self.time_horizon and not end_flag:
            if np.linalg.norm(waypoint[0:2]-current_pose[0:2]) < self.delta_pos:
                end_flag = True
                continue
            t += self.time_step_simulation
            heading_delta = np.clip(ssa(heading_d,current_heading), -self.heading_delta_max, self.heading_delta_max)
            velocity_delta = np.clip(velocity_d-current_velocity, -self.vel_delta_max, self.vel_delta_max)

            current_heading += self.time_step_simulation*heading_delta
            current_velocity += self.time_step_simulation*velocity_delta

            current_pose[0] += self.time_step_simulation*np.cos(current_heading)*current_velocity
            current_pose[1] += self.time_step_simulation*np.sin(current_heading)*current_velocity
            current_pose[2] = current_heading % (2*np.pi)

            trajectory.append(current_pose.copy())

        return trajectory

    def update_velocity_commands(self, velocity_commands):
        self.velocity_commands = velocity_commands

    def update_heading_commands(self, heading_commands):
        self.heading_commands = heading_commands

    def plot_trajectory(self, distance_grid, trajectory):
        fig, axis = plt.subplots(1)
        axis.imshow(distance_grid["grid"].transpose(), origin="lower", extent=[distance_grid["lower_x"], distance_grid["lower_x"] + distance_grid["grid"].shape[0]*distance_grid["res"],
                                                   distance_grid["lower_y"], distance_grid["lower_y"] + distance_grid["grid"].shape[0]*distance_grid["res"]])
        axis.plot([t[0] for t in trajectory],[t[1] for t in trajectory],"-o",color="red", markersize=0.1)
        plt.show()