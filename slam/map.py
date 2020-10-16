import numpy as np
from math import sin, cos
import time
import matplotlib.pyplot as plt
from logger.logger import Logger
from sensor.lidar_sensor import LidarSensor
from matplotlib.animation import FuncAnimation
from matplotlib import animation

class SLAM_map():
    """
    Class for tracking a single SLAM map. The map uses occupancy grid representation of the environment. The cells are
    updated based on probabilistic model.

    Attributes:
         size_x: size of map in X-direction [m]
         size_y: size of map in Y-direction [m]
         res: size of each cell [m]
         prior: the prior probability of a cell being occupied [0,1]
         grid: numpy array representing the probability of the cell being occupied [np.array(x,y)]


    """

    def __init__(self, size_x=1000, size_y=1000, res = 0.1, p_prior=0.5, p_occupied=0.7, p_free=0.3):
        self.size_x = size_x+2
        self.size_y = size_y+2
        self.res = res

        self.max_range = 4

        self.log_free = np.log(p_free / (1 - p_free))
        self.log_occupied = np.log(p_occupied / (1 - p_occupied))
        self.log_prior = np.log(p_prior / (1 - p_prior))

        self.center_x = np.int(self.size_x/2)
        self.center_y = np.int(self.size_y/2)

        self.log_prob_map = np.ones((self.size_x, self.size_y))*self.log_prior
        self.grid_position_m = np.array(
            [np.tile(np.arange(0, self.size_x * self.res, self.res)[:, None], (1, self.size_y))-self.center_x*self.res,
             np.tile(np.arange(0, self.size_y * self.res, self.res)[:, None].T, (self.size_x, 1))-self.center_y*self.res])
        self.alpha = 0.2
        self.beta = 2*np.pi/180

    def integrate_scan(self, pose, measurments, ray_vectors):
        ray_vectors = rot_matrix_2d(pose[2]) @ ray_vectors[0:2,:]
        for i in range(measurments.size):
            meas = measurments[i]
            if meas > self.max_range + 0.1 and meas != np.inf:
                print("Weird measurments")
                continue
            ray = ray_vectors[0:2,i].reshape(2,1)
            if meas == np.inf:
                end_point = pose[0:2] + ray * self.max_range
            else:
                end_point = pose[0:2] + ray * meas

            end_point_cell = self.world_coordinate_to_grid_cell(end_point)
            if self.cell_in_grid(end_point_cell) and meas != np.inf:
                self.log_prob_map[end_point_cell[0],end_point_cell[1]] += self.log_occupied
            free_cells = self.grid_traversal(pose[0:2], end_point)
            for cell in free_cells:
                if self.cell_in_grid(cell) and np.any(cell != end_point_cell):
                    self.log_prob_map[cell[0],cell[1]] += self.log_free

    def world_coordinate_to_grid_cell(self,coord):
        cell = np.array([np.int(np.floor(coord[0]/self.res)) + self.center_x, np.int(np.floor(coord[1]/self.res))+self.center_y])
        return cell

    def cell_in_grid(self, cell):
        return not (np.any(cell < 0) or np.any(cell >= np.array([self.size_x,self.size_y])))

    def grid_traversal(self, start_point, end_point):
        visited_cells = []
        current_cell = self.world_coordinate_to_grid_cell(start_point)
        last_cell = self.world_coordinate_to_grid_cell(end_point)
        ray = (end_point-start_point)
        ray = ray/np.linalg.norm(ray)

        stepX = 1 if ray[0] >= 0 else -1
        stepY = 1 if ray[1] >= 0 else -1

        next_cell_boundary_x = (current_cell[0]-self.center_x + stepX)*self.res
        next_cell_boundary_y = (current_cell[1]-self.center_y + stepY)*self.res

        tmaxX = (next_cell_boundary_x-start_point[0])/ray[0] if ray[0] != 0 else np.inf
        tmaxY = (next_cell_boundary_y-start_point[1])/ray[1] if ray[1] != 0 else np.inf

        tdeltaX = self.res/ray[0]*stepX if ray[0] != 0 else np.inf
        tdeltaY = self.res/ray[1]*stepY if ray[1] != 0 else np.inf

        neg_ray = False
        diff = np.zeros(2,np.int)
        if current_cell[0] != last_cell[0] and ray[0]<0:
            diff[0] -= 1
            neg_ray = True
        if current_cell[1] != last_cell[1] and ray[1]<0:
            diff[1] -= 1
            neg_ray = True
        if neg_ray:
            visited_cells.append(current_cell.copy())
            current_cell += diff

        while np.any(current_cell != last_cell) and self.cell_in_grid(current_cell):
            visited_cells.append(current_cell.copy())
            if tmaxX <= tmaxY:
                current_cell[0] += stepX
                tmaxX += tdeltaX
            else:
                current_cell[1] += stepY
                tmaxY += tdeltaY
        return visited_cells

    def convert_grid_to_prob(self):
        exp_grid = np.exp(self.log_prob_map)
        return exp_grid/(1+exp_grid)

def rot_matrix_2d(theta):
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    r = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return r

def test():
    file = "test_lidar.json"
    log = Logger()
    log.load_from_file("test_lidar.json")
    measurements = log.get_drone_sensor_measurements("0",0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0",0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy,sensor_specs.sensor_attitude_bdy,num_rays=72)
    rays = sensor.ray_vectors

    map = SLAM_map(size_x=400,size_y=400)
    maps = []
    for i in range(0,states["steps"],10):
        meas = measurements["measurements"][i]
        pose = states["states"][i][[0,1,5]]
        map.integrate_scan(pose,meas,rays)
        if i % 20 == 0:
            maps.append({"map":map.convert_grid_to_prob(),"pose":pose,"time":i*states["step_length"]})

    def visualize():
        fig, axis = plt.subplots(1)
        drone = plt.Circle((states["states"][0][0] / map.res + map.center_x, states["states"][0][1] / map.res + map.center_y), radius=1,color="red")
        axis.add_patch(drone)
        def animate(t):
            axis.axis("equal")
            axis.set_title("Time: {:.2f}".format(t["time"]))
            axis.imshow(t["map"].transpose(),"Greys",origin="lower")
            drone.set_center([t["pose"][0] / map.res + map.center_x, t["pose"][1] / map.res + map.center_y])
        anim = FuncAnimation(fig, animate, frames=maps, repeat=False, blit=False, interval=500)

        plt.draw()
        plt.show()
    visualize()

if __name__ == "__main__":
    test()