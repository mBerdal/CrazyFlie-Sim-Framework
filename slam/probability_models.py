from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d
from utils.rotation_utils import ssa, rot_matrix_2d
from utils.misc import gaussian_kernel_2d
from math import floor
import matplotlib.pyplot as plt

class ProbModel(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def likelihood(self,new_pose, prev_pose, u):
        pass

    @abstractmethod
    def sample(self, prev_pose, u):
        pass


class OdometryModel(ProbModel):

    def __init__(self, **kwargs):
        self.alpha1 = kwargs.get("alpha1", 0.15)
        self.alpha2 = kwargs.get("alpha2", 0.15)
        self.alpha3 = kwargs.get("alpha3", 0.25)
        self.alpha4 = kwargs.get("alpha4", 0.25)

    def likelihood(self,new_pose, prev_pose, u):
        od_prev = u[1,:]
        od_new = u[0,:]

        delta_rot1_od = ssa(np.arctan2(od_new[1]-od_prev[1],od_new[0]-od_new[0]),od_prev[2])
        delta_trans_od = np.linalg.norm(od_prev[0:2]-od_new[0:2])
        delta_rot2_od = ssa(od_new[2],(od_prev[2]-delta_rot1_od)%(2*np.pi))

        delta_rot1_pose = ssa(np.arctan2(new_pose[1] - prev_pose[1], new_pose[0] - prev_pose[0]),prev_pose[2])
        delta_trans_pose = np.linalg.norm(prev_pose[0:2] - new_pose[0:2])
        delta_rot2_pose = ssa(new_pose[2], (prev_pose[2] - delta_rot1_pose)%(2*np.pi))

        std1 = self.alpha1*delta_rot1_od**2+ self.alpha2*delta_trans_od**2 + 1e-2
        err1 = ssa(delta_rot1_pose,delta_rot1_od)
        p1 = norm.pdf(err1,scale=std1)

        std2 = self.alpha3*delta_trans_od**2+self.alpha4*(delta_rot1_od**2 + delta_rot2_od**2) + 1e-2
        err2 = delta_trans_pose-delta_trans_od
        p2 = norm.pdf(err2,scale=std2)

        err3 = ssa(delta_rot2_pose,delta_rot2_od)
        p3 = norm.pdf(err3,scale=std1)

        return p1*p2*p3

    def sample(self, prev_pose, u):
        pose = np.zeros([3,1])

        od_prev = u[0, :]
        od_new = u[1, :]

        delta_rot1 = np.arctan2(od_new[1] - od_prev[1], od_new[0] - od_new[0])-od_prev[2]
        delta_trans = np.linalg.norm(od_prev[0:1] - od_new[0:1])
        delta_rot2 = od_new[2] - od_prev[2]-delta_rot1

        delta_rot1_hat = delta_rot1 - norm.rvs(scale=(self.alpha1*delta_rot1**2 + self.alpha2*delta_trans**2))
        delta_trans_hat = delta_trans - norm.rvs(scale=(self.alpha3*delta_trans**2 + self.alpha4*(delta_rot1**2 + delta_rot2**2)))
        delta_rot2_hat = delta_rot2 - norm.rvs(scale=(self.alpha1*delta_rot1**2 +self.alpha2*delta_trans**2))

        pose[0] = prev_pose[0] + delta_trans_hat*np.cos(prev_pose[2]+delta_rot1_hat)
        pose[1] = prev_pose[1] + delta_trans_hat*np.sin(prev_pose[2]+delta_rot1_hat)
        pose[2] = (prev_pose[2] + delta_rot2_hat + delta_rot1_hat) % (2*np.pi)

        return pose


class ObservationModel:
    def __init__(self, ray_vectors, **kwargs):
        self.ray_vectors = ray_vectors
        self.skip_rays = kwargs.get("skip_rays",4)
        self.ray_vectors = self.ray_vectors[:,0::self.skip_rays]
        self.occ_threshold = kwargs.get("occ_threshold",0.7)
        self.eps = 0.01

        self.like_field = None
        self.lower_x = 0
        self.lower_y = 0
        self.map_res = None

        self.sigma = kwargs.get("sigma",0.5)
        self.p_unknown = 0.2

        self.max_sensor_range = kwargs.get("max_sensor_range", 10)
        self.padding_dist = 2.0
        self.kernel_size = kwargs.get("kernel_size", 3)
        self.kernel = gaussian_kernel_2d(self.kernel_size, self.sigma)

    def likelihood(self, pose, measurements):
        q = 1
        pose = pose.squeeze()
        rays = rot_matrix_2d(pose[2]) @ self.ray_vectors[0:2, :]
        measurements = measurements[0::self.skip_rays]
        for ind, meas in enumerate(measurements):
            if meas == np.inf:
                continue
            else:
                end_point = pose[0:2] + rays[:, ind] * meas
            cell_x = int(floor((end_point[0] - self.lower_x) / self.map_res))
            cell_y = int(floor((end_point[1] - self.lower_y) / self.map_res))

            if cell_x < 0 or cell_x >= self.size_x or cell_y < 0 or cell_y >= self.size_y:
                continue
            q = q*(self.like_field[cell_x, cell_y] + self.eps)
        return q

    def likelihood_vectorized(self, pose, measurements):
        pose = pose.squeeze().reshape(3,1)
        rays = rot_matrix_2d(pose[2]) @ self.ray_vectors[0:2,:]
        measurements = measurements[0::self.skip_rays]

        end_point = pose[0:2] + np.multiply(rays,measurements)
        cells_x = np.floor((end_point[0,:]-self.lower_x)/self.map_res)
        cells_y = np.floor((end_point[1,:]-self.lower_y)/self.map_res)

        unvalid_ind = ~((cells_x < 0) | (cells_x >= self.size_x) | (cells_y < 0) | (cells_x >= self.size_y))

        cells_x = cells_x[unvalid_ind].astype(int)
        cells_y = cells_y[unvalid_ind].astype(int)
        likelihood = self.like_field[cells_x,cells_y]
        return np.sum(likelihood)

    def compute_likelihood_field(self, initial_pose, map):
        lower_x = np.int(
            np.floor((initial_pose[0] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_x
        lower_y = np.int(
            np.floor((initial_pose[1] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_y

        upper_x = np.int(
            np.floor((initial_pose[0] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_x + 1
        upper_y = np.int(
            np.floor((initial_pose[1] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_y + 1

        lower_x = max([lower_x, 0])
        lower_y = max([lower_y, 0])

        upper_x = min([upper_x, map.size_x])
        upper_y = min([upper_y, map.size_y])

        prob_field = map.log_prob_map[lower_x:upper_x, lower_y:upper_y].copy()

        binary_field = prob_field > np.log(self.occ_threshold / (1 - self.occ_threshold))
        like_field = convolve2d(binary_field, self.kernel, mode="same")
        unknown_area = prob_field == 0
        like_field[unknown_area] = self.p_unknown
        self.like_field = like_field
        self.lower_x = (lower_x - map.center_x) * map.res
        self.lower_y = (lower_y - map.center_y) * map.res
        self.map_res = map.res
        self.size_x = like_field.shape[0]
        self.size_y = like_field.shape[1]

    def compute_likelihood_field_dist(self,initial_pose,map):
        lower_x = np.int(
            np.floor((initial_pose[0] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_x
        lower_y = np.int(
            np.floor((initial_pose[1] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_y

        upper_x = np.int(
            np.floor((initial_pose[0] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_x + 1
        upper_y = np.int(
            np.floor((initial_pose[1] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_y + 1

        lower_x = max([lower_x, 0])
        lower_y = max([lower_y, 0])

        upper_x = min([upper_x, map.size_x])
        upper_y = min([upper_y, map.size_y])

        prob_field = map.log_prob_map[lower_x:upper_x, lower_y:upper_y].copy()

        binary_field = prob_field > np.log(self.occ_threshold / (1 - self.occ_threshold))
        occupied_cells_x, occupied_cells_y = np.where(binary_field)

        dist_field = np.inf * np.ones(prob_field.shape)
        dist_field[occupied_cells_x, occupied_cells_y] = 0
        x = np.arange(0, dist_field.shape[1]) * map.res
        y = np.arange(0, dist_field.shape[0]) * map.res
        xx, yy = np.meshgrid(x, y)
        for i in range(occupied_cells_x.shape[0]):
            cell_x = occupied_cells_x[i]
            cell_y = occupied_cells_y[i]
            dist_occupied = np.sqrt((xx - xx[cell_x, cell_y]) ** 2 + (yy - yy[cell_x, cell_y]) ** 2)
            dist_field = np.minimum(dist_field, dist_occupied)
        dist_field = np.exp(-dist_field ** 2 / (2 * self.sigma ** 2)) / (np.sqrt(2 * np.pi))
        unknown_area = (prob_field == 0)
        #dist_field[unknown_area] = self.p_unknown
        self.like_field = dist_field
        self.lower_x = (lower_x - map.center_x) * map.res
        self.lower_y = (lower_y - map.center_y) * map.res
        self.map_res = map.res
        self.size_x = dist_field.shape[0]
        self.size_y = dist_field.shape[1]

    def visualize_likelihood(self,cells_x,cells_y):
        plt.figure()
        plt.imshow(self.like_field.T, origin="lower")
        plt.plot(cells_x,cells_y,"o",markersize=1,color="red")
        plt.colorbar()
        plt.show()