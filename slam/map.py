import numpy as np
from math import sin, cos
import time
import matplotlib.pyplot as plt

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

    def __init__(self, size_x=1000, size_y=1000, res = 0.1, p_prior=0.5, p_occupied=0.8, p_free=0.2):
        self.size_x = size_x+2
        self.size_y = size_y+2
        self.res = res

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

        for meas, ray in zip(measurments,ray_vectors):
            if meas == np.inf:
                continue
            end_point = pose[0:2] + rot_matrix_2d(pose[2]) @ ray * meas
            end_point_cell = self.world_coordinate_to_grid_cell(end_point)
            if self.cell_in_grid(end_point_cell):
                self.log_prob_map[end_point_cell[0],end_point_cell[1]] += self.log_occupied
            free_cells = self.grid_traversal(pose[0:2], end_point)
            for cell in free_cells:
                self.log_prob_map[cell[0],cell[1]] += self.log_free

    def integrate_scan_v2(self,pose, z):
        dx = self.grid_position_m.copy()  # A tensor of coordinates of all cells
        dx[0, :, :] -= pose[0]  # A matrix of all the x coordinates of the cell
        dx[1, :, :] -= pose[1]  # A matrix of all the y coordinates of the cell
        theta_to_grid = np.arctan2(dx[1, :, :], dx[0, :, :]) - pose[2]  # matrix of all bearings from robot to cell

        # Wrap to +pi / - pi
        theta_to_grid[theta_to_grid > np.pi] -= 2. * np.pi
        theta_to_grid[theta_to_grid < -np.pi] += 2. * np.pi

        dist_to_grid = np.linalg.norm(dx, axis=0)  # matrix of L2 distance to all cells from robot

        # For each laser beam
        for z_i in z:
            r = z_i[0]  # range measured
            b = z_i[1]  # bearing measured

            # Calculate which cells are measured free or occupied, so we know which cells to update
            # Doing it this way is like a billion times faster than looping through each cell (because vectorized numpy is the only way to numpy)
            free_mask = (np.abs(theta_to_grid - b) <= self.beta / 2.0) & (dist_to_grid < (r - self.alpha / 2.0))
            occ_mask = (np.abs(theta_to_grid - b) <= self.beta / 2.0) & (np.abs(dist_to_grid - r) <= self.alpha / 2.0)

            # Adjust the cells appropriately
            self.log_prob_map[occ_mask] += self.log_occupied
            self.log_prob_map[free_mask] += self.log_free

    def world_coordinate_to_grid_cell(self,coord):
        cell = np.array([np.int(coord[0]/self.res) + self.center_x, np.int(coord[1]/self.res)+self.center_y])
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

        while np.any(current_cell != last_cell) and self.cell_in_grid(current_cell):
            visited_cells.append(current_cell.copy())
            if tmaxX < tmaxY:
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
    num_beams = 36
    map = SLAM_map()
    meas = []
    meas_v2 = []
    rays = []
    for i in range(num_beams):
        dist = np.random.uniform(0,4)
        bearing = np.random.uniform(0,2*np.pi)
        meas_v2.append(np.array([dist,bearing]))
        meas.append(dist)
        ray = np.array([sin(bearing),cos(bearing)])
        rays.append(ray/np.linalg.norm(ray))
    pose = np.array([0,0,0])

    prior = map.convert_grid_to_prob()

    start = time.time()
    map.integrate_scan(pose,meas,rays)
    end = time.time()

    posterior = map.convert_grid_to_prob()
    plt.figure()
    plt.imshow(posterior, 'Greys')


    map = SLAM_map()
    start_v2 = time.time()
    map.integrate_scan_v2(pose,meas_v2)
    end_v2 = time.time()

    posterior = map.convert_grid_to_prob()
    plt.figure()
    plt.imshow(posterior, 'Greys')
    plt.show()

    print("Time version 1:",end-start)
    print("Time version 2:",end_v2-start_v2)

if __name__ == "__main__":
    test()