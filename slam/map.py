import numpy as np
import copy
from utils.rotation_utils import rot_matrix_2d
from math import floor


class SLAM_map:
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

    def __init__(self, **kwargs):
        self.size_x = kwargs.get("size_x",400)
        self.size_y = kwargs.get("size_y",400)
        self.res = kwargs.get("res",0.1)

        self.max_range = kwargs.get("max_range",10)

        self.p_free = kwargs.get("p_free",0.4)
        self.p_occupied = kwargs.get("p_occupied",0.8)
        self.p_prior = kwargs.get("p_prior",0.5)
        self.log_free = np.log(self.p_free / (1 - self.p_free))
        self.log_occupied = np.log(self.p_occupied / (1 - self.p_occupied))
        self.log_prior = np.log(self.p_prior / (1 - self.p_prior))

        self.center_x = int(floor(self.size_x/2))
        self.center_y = int(floor(self.size_y/2))

        self.log_prob_map = np.ones((self.size_x, self.size_y),dtype=np.float32)*self.log_prior

    def integrate_scan(self, pose, measurments, ray_vectors):
        ray_vectors = rot_matrix_2d(pose[2]) @ ray_vectors[0:2, :]
        for i in range(measurments.size):
            meas = measurments[i]
            ray = ray_vectors[0:2, i].reshape(2,1)
            if meas == np.inf:
                end_point = pose[0:2] + ray * self.max_range
            else:
                end_point = pose[0:2] + ray * meas

            end_point_cell = self.world_coordinate_to_grid_cell(end_point)
            if meas != np.inf:
                if not (end_point_cell[0] < 0 or end_point_cell[0] > self.size_x or end_point_cell[1] < 0 or end_point_cell[1] >= self.size_y):
                    self.log_prob_map[end_point_cell[0],end_point_cell[1]] += self.log_occupied
            free_cells = self.grid_traversal(pose[0:2], end_point)
            for cell in free_cells:
                if not (cell[0] < 0 or cell[0] > self.size_x or cell[1] < 0 or cell[1] >= self.size_y):
                    if cell[0] == end_point_cell[0] and cell[1] == end_point_cell[1]:
                        print("Error grid traversal")
                    self.log_prob_map[cell[0],cell[1]] += self.log_free

    def world_coordinate_to_grid_cell(self,coord):
        cell = np.array([int(floor(coord[0]/self.res)) + self.center_x, int(floor(coord[1]/self.res))+self.center_y])
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
        if current_cell[0] != last_cell[0] and ray[0] < 0:
            diff[0] -= 1
            neg_ray = True
        if current_cell[1] != last_cell[1] and ray[1] < 0:
            diff[1] -= 1
            neg_ray = True
        if neg_ray:
            visited_cells.append(current_cell.copy())
            current_cell += diff

        while current_cell[0] != last_cell[0] or current_cell[1] != last_cell[1]:
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

    def init_plot(self, axis):
        im = axis.imshow(self.convert_grid_to_prob().transpose(),"Greys",origin="lower",
                   extent=[-self.size_x/2*self.res,self.size_x/2*self.res,-self.size_y/2*self.res,self.size_y/2*self.res])
        return im

    def update_plot(self, im):
        im.set_data(self.convert_grid_to_prob().transpose())
        im.autoscale()
        return im

    def __deepcopy__(self, memodict={}):
        m = SLAM_map(size_x=self.size_x,size_y=self.size_y,res=self.res)
        m.log_prob_map = copy.deepcopy(self.log_prob_map)
        return m
