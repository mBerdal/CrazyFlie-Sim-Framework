import numpy as np
from scipy.signal import convolve2d
from utils.misc import gaussian_kernel_2d, gaussian_kernel_2d_v2
from utils.rotation_utils import rot_matrix_2d
from math import floor

class ScanMatcher():
    """
    Class for performing scan matching between the current measurements and the most recent map.

    Uses a greedy steepest descent approach to find the best match.

    """
    def __init__(self, **kwargs):
        self.max_sensor_range = kwargs.get("max_sensor_range",10)
        self.padding_dist = 1.0
        self.skip_rays = kwargs.get("skip_rays",4)

        self.sigma = kwargs.get("sigma",0.3)
        self.p_unknown = 0.3

        self.initial_step = kwargs.get("step",1)
        self.max_iterations = kwargs.get("max_iterations",10)

        self.delta_x = kwargs.get("delta_x",0.1)
        self.delta_y = kwargs.get("delta_y",0.1)
        self.delta_theta = kwargs.get("delta_theta",np.pi*2/180)

        self.perturbations = [np.array([-self.delta_x, 0, 0]), np.array([self.delta_x, 0, 0]),
                              np.array([0, -self.delta_y, 0]), np.array([0, self.delta_y, 0]),
                              np.array([0, 0, -self.delta_theta]), np.array([0, 0, self.delta_theta])]


        self.kernel_size = kwargs.get("kernel_size",2)
        self.kernel = gaussian_kernel_2d(self.kernel_size, self.sigma)

        self.occ_threshold = kwargs.get("occ_threshold",0.7)

    def scan_match(self, rays, meas, initial_pose, map):
        map_res = map.res
        like_field, lower_x, lower_y = self.compute_likelihood_field(map, initial_pose)
        step = self.initial_step
        iterations = 0
        current_pose = initial_pose.copy()
        best_score = self.score_scan(rays, meas, current_pose, like_field, lower_x, lower_y, map_res)
        initial_score = best_score
        best_pose = initial_pose.copy()
        while iterations < self.max_iterations:
            best_perturbation = np.zeros(3)
            current_score = best_score
            for pet in self.perturbations:
                tmp_pose = (current_pose + step*pet.reshape(3,1)).copy()
                tmp_score = self.score_scan(rays, meas, tmp_pose, like_field, lower_x, lower_y, map_res)
                if tmp_score > current_score:
                    current_score = tmp_score
                    best_perturbation = pet
            if current_score > best_score:
                best_score = current_score
                best_pose = (current_pose + step*best_perturbation.reshape(3,1)).copy()
            else:
                step = step/2
                iterations += 1
            current_pose = best_pose.copy()
        best_pose[2] = best_pose[2] % (2*np.pi)
        diff = best_score-initial_score
        return best_pose.reshape(3, 1), diff

    def score_scan(self, rays, measurements, pose, like_field, lower_x, lower_y, map_res):
        n = like_field.shape[0]
        m = like_field.shape[1]

        pose = pose.squeeze().reshape(3, 1)
        rays = rot_matrix_2d(pose[2]) @ rays[0:2,0::self.skip_rays]
        measurements = measurements[0::self.skip_rays]

        end_point = pose[0:2] + np.multiply(rays, measurements)
        cells_x = np.floor((end_point[0, :] - lower_x) / map_res)
        cells_y = np.floor((end_point[1, :] - lower_y) / map_res)

        unvalid_ind = ~((cells_x < 0) | (cells_x >= n) | (cells_y < 0) | (cells_y >= m))

        cells_x = cells_x[unvalid_ind].astype(int)
        cells_y = cells_y[unvalid_ind].astype(int)
        likelihood = like_field[cells_x, cells_y]
        return np.sum(likelihood)

    def compute_likelihood_field(self, map, initial_pose):
        #Approximates the likelihood field using convolution on the binary occupancy grid with a gaussian kernel
        lower_x = np.int((initial_pose[0] - self.max_sensor_range - self.padding_dist)/map.res) + map.center_x
        lower_y = np.int((initial_pose[1] - self.max_sensor_range - self.padding_dist)/map.res) + map.center_y

        upper_x = np.int((initial_pose[0] + self.max_sensor_range + self.padding_dist)/map.res) + map.center_x + 1
        upper_y = np.int((initial_pose[1] + self.max_sensor_range + self.padding_dist)/map.res) + map.center_y + 1

        prob_field = map.log_prob_map[lower_x:upper_x, lower_y:upper_y].copy()

        binary_field = prob_field > np.log(self.occ_threshold/(1-self.occ_threshold))

        like_field = convolve2d(binary_field, self.kernel, mode="same")

        return like_field, (lower_x - map.center_x)*map.res, (lower_y-map.center_y)*map.res

    def compute_distance_field(self, map, initial_pose):
        #Computes the likelihood field using the distance to nearest occupied cell
        lower_x = np.int(np.floor((initial_pose[0] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_x
        lower_y = np.int(np.floor((initial_pose[1] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_y

        upper_x = np.int(np.floor((initial_pose[0] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_x + 1
        upper_y = np.int(np.floor((initial_pose[1] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_y + 1

        lower_x = max([lower_x, 0])
        lower_y = max([lower_y, 0])

        upper_x = min([upper_x, map.size_x])
        upper_y = min([upper_y, map.size_y])

        prob_field = map.log_prob_map[lower_x:upper_x, lower_y:upper_y].copy()

        binary_field = prob_field > np.log(self.occ_threshold / (1 - self.occ_threshold))
        occupied_cells_x, occupied_cells_y = np.where(binary_field)

        dist_field = np.inf*np.ones(prob_field.shape)
        dist_field[occupied_cells_x,occupied_cells_y] = 0
        x = np.arange(0,dist_field.shape[1])*map.res
        y = np.arange(0,dist_field.shape[0])*map.res
        xx, yy = np.meshgrid(x,y)
        for i in range(occupied_cells_x.shape[0]):
            cell_x = occupied_cells_x[i]
            cell_y = occupied_cells_y[i]
            dist_occupied = np.sqrt((xx-xx[cell_x,cell_y])**2 + (yy-yy[cell_x,cell_y])**2)
            dist_field = np.minimum(dist_field,dist_occupied)
        dist_field = np.exp(-dist_field**2/(2*self.sigma**2))/(np.sqrt(2*np.pi))
        return dist_field, (lower_x - map.center_x)*map.res, (lower_y-map.center_y)*map.res
