from slam.map import SLAM_map
import numpy as np
from queue import Queue
from slam.map import SLAM_map
import time
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

class ScanMatcher():

    def __init__(self, max_iterations=5, delta_x=0.1, delta_y=0.1, delta_theta=np.pi*10/180, max_sensor_range=5.0, step=1):
        self.max_sensor_range = max_sensor_range
        self.padding_dist = 2.0

        self.sigma_hit = 0.1
        self.hit_weight = 1
        self.z_random = 0.001
        self.z_max = 0.2

        self.max_iterations = max_iterations

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_theta = delta_theta
        self.perturbations = [np.array([-self.delta_x, 0, 0]), np.array([self.delta_x, 0, 0]),
                              np.array([0, -self.delta_y, 0]), np.array([0, self.delta_y, 0]),
                              np.array([0, 0, -self.delta_theta]), np.array([0, 0, self.delta_theta])]

        self.initial_step = step

        self.kernal = gaussian_kernal_2d(5,0.5)
        self.occ_threshold = 0.7

    def scan_match(self, rays, meas, initial_pose, map):
        map_res = map.res
        like_field, lower_x, lower_y = self.compute_likelihood_field(map, initial_pose)
        step = self.initial_step
        iterations = 0
        current_pose = initial_pose
        best_score = self.score_scan(rays, meas, current_pose, like_field, lower_x, lower_y, map_res)
        best_pose = initial_pose
        while iterations < self.max_iterations:
            best_perturbation = np.zeros(3)
            current_score = best_score
            for pet in self.perturbations:
                tmp_pose = current_pose + step*pet.reshape(3,1)
                tmp_score = self.score_scan(rays, meas, tmp_pose, like_field, lower_x, lower_y, map_res)
                if tmp_score > current_score:
                    current_score = tmp_score
                    best_perturbation = pet
            if current_score > best_score:
                best_score = current_score
                best_pose = current_pose + step*best_perturbation.reshape(3,1)
            else:
                step = step/2
            iterations += 1
            current_pose = best_pose
        return best_pose

    def score_scan(self, rays, measurements, pose, like_field, lower_x, lower_y, map_res):
        q = 0
        pose = pose.squeeze()
        rays = rot_matrix_2d(pose[2]) @ rays[0:2,:]
        for ind, meas in enumerate(measurements):
            if meas == np.inf:
                continue
            else:
                end_point = pose[0:2] + rays[:,ind]*meas
            cell_x = np.int(np.floor((end_point[0] -lower_x)/map_res))
            cell_y = np.int(np.floor((end_point[1] -lower_y)/map_res))
            try:
                q = q+(self.hit_weight*like_field[cell_x, cell_y] + self.z_random/self.z_max)
            except:
                pass
        return q

    def compute_likelihood_field(self, map, initial_pose):
        lower_x = np.int((initial_pose[0] - self.max_sensor_range - self.padding_dist)/map.res) + map.center_x
        lower_y = np.int((initial_pose[1] - self.max_sensor_range - self.padding_dist)/map.res) + map.center_y

        upper_x = np.int((initial_pose[0] + self.max_sensor_range + self.padding_dist)/map.res) + map.center_x + 1
        upper_y = np.int((initial_pose[1] + self.max_sensor_range + self.padding_dist)/map.res) + map.center_y + 1

        prob_field = map.log_prob_map[lower_x:upper_x, lower_y:upper_y].copy()

        binary_field = prob_field > np.log(self.occ_threshold/(1-self.occ_threshold))

        like_field = convolve2d(binary_field, self.kernal, mode="same")

        unknown_area = prob_field == 0
        like_field[unknown_area] = 1/self.max_sensor_range**2

        return like_field, (lower_x - map.center_x)*map.res, (lower_y-map.center_y)*map.res

def gaussian_kernal_2d(size, sigma):
    xx = np.arange(-size,size+1,1)
    yy = np.arange(-size,size+1,1)
    x_grid, y_grid = np.meshgrid(xx,yy)
    kernal = np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernal

def rot_matrix_2d(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    r = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    return r

def test():
    num_beams = 36
    scan = ScanMatcher()
    map = SLAM_map()


    meas = []
    rays = []
    for i in range(num_beams):
        dist = np.random.uniform(0, 4)
        bearing = np.random.uniform(-np.pi, np.pi)
        meas.append(dist)
        ray = np.array([np.sin(bearing), np.cos(bearing)])
        rays.append(ray / np.linalg.norm(ray))
    pose = np.array([0, 0, 0])
    map.integrate_scan(pose,meas,rays)
    rays = np.array(rays).transpose()
    pose = np.array([0,1,0])
    start = time.time()
    matched_pose = scan.scan_match(rays,meas,pose,map)
    end = time.time()

    print("Time: {:.6f}".format(end-start))
    print("Mathced Pose:",matched_pose)
if __name__ == "__main__":
    test()