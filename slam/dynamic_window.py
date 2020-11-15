import math
import numpy as np
import matplotlib.pyplot as plt
from utils.rotation_utils import ssa

class DynamicWindow:

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = 0.0  # [m/s]
        self.max_yaw_rate = 360 * math.pi / 180.0  # [rad/s]
        self.max_accel = 3  # [m/ss]
        self.max_delta_yaw_rate = 1080 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.05  # [m/s]
        self.yaw_rate_resolution = 2 * math.pi / 180.0  # [rad/s]
        self.dt = 0.05  # [s] Time tick for motion prediction
        self.predict_time = 2.0  # [s]
        self.to_goal_cost_gain = 5.0
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.1
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked

    def predict_trajectory(self, x_init, v, y):
        """
        predict trajectory with an input
        """

        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            x = self.motion(x, [v, y])
            trajectory = np.hstack((trajectory, x))
            time += self.dt
        return trajectory

    def motion(self,x, u):
        """
        motion model
        """
        x[2] += u[1] * self.dt
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    def calc_dynamic_window(self, x):
        """
        calculation dynamic window based on current state x
        """

        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
              -self.max_yaw_rate, self.max_yaw_rate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
              x[3] + self.max_accel * self.dt,
              x[4] - self.max_delta_yaw_rate * self.dt,
              x[4] + self.max_delta_yaw_rate * self.dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
              max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

        return dw

    def calc_control(self, state, goal, distance_grid):
        """
        calculation final input with dynamic window
        """

        state_init = state[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        dw = self.calc_dynamic_window(state)

        # evaluate all trajectory with sampled input in dynamic window
        for v in np.arange(dw[0], dw[1]+self.v_resolution, self.v_resolution):
            for y in np.arange(dw[2], dw[3]+self.yaw_rate_resolution, self.yaw_rate_resolution):

                trajectory = self.predict_trajectory(state_init, v, y)
                # calc cost
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - trajectory[3, -1])
                ob_cost = self.obstacle_cost_gain * self.calc_obstacle_cost(trajectory, distance_grid)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.robot_stuck_flag_cons \
                            and abs(state[3]) < self.robot_stuck_flag_cons:
                        # to ensure the robot do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.max_delta_yaw_rate
        return best_u

    def visualize_trajectory(self, trajectory, dist_grid):
        plt.figure()
        extent = [dist_grid["lower_x"], dist_grid["lower_x"]+dist_grid["grid"].shape[0]*dist_grid["res"],
                  dist_grid["lower_y"], dist_grid["lower_y"]+dist_grid["grid"].shape[1]*dist_grid["res"]]
        plt.imshow(dist_grid["grid"].transpose(), origin="lower", extent=extent)
        plt.plot(trajectory[0,:],trajectory[1,:],color="red")
        plt.pause(1)
        plt.close(plt.gcf())


    def calc_obstacle_cost(self, trajectory, distance_grid):
        """
        calc obstacle cost inf: collision
        """
        dist = []
        for i in range(trajectory.shape[1]):
            x_cell = np.int(np.floor((trajectory[0,i] - distance_grid["lower_x"]) / distance_grid["res"]))
            y_cell = np.int(np.floor((trajectory[1,i] - distance_grid["lower_y"]) / distance_grid["res"]))
            try:
                distance = distance_grid["grid"][x_cell, y_cell]
                dist.append(distance)
            except:
                pass
        min_dist = min(dist)
        if min_dist < 0.10:
            return 1/min_dist
        min_dist = max(min_dist,0.2)
        velocity = max(trajectory[3,-1],0.3)
        return 1.0*velocity / min_dist  # OK


    def calc_to_goal_cost(self, trajectory, goal):
        """
            calc to goal cost with angle difference
        """

        dx = goal[0] - trajectory[0, -1]
        dy = goal[1] - trajectory[1, -1]
        error_angle = math.atan2(dy, dx)
        cost_angle = ssa(error_angle, trajectory[2, -1])
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle))) #+ math.sqrt(dx**2+dy**2)
        return cost

