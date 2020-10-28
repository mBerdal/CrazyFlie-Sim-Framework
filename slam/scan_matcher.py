from slam.map import SLAM_map
import numpy as np
from queue import Queue
from slam.map import SLAM_map
import time
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from utils.misc import gaussian_kernel_2d
from utils.rotation_utils import rot_matrix_2d
from logger.logger import Logger
from sensor.lidar_sensor import LidarSensor

class ScanMatcher():

    def __init__(self, **kwargs):
        self.max_sensor_range = kwargs.get("max_sensor_range",4)
        self.padding_dist = 2.0

        self.sigma = kwargs.get("sigma",0.5)
        self.p_unknown = 0.3

        self.initial_step = kwargs.get("step",1)
        self.max_iterations = kwargs.get("max_iterations",10)

        self.delta_x = kwargs.get("delta_x",0.1)
        self.delta_y = kwargs.get("delta_y",0.1)
        self.delta_theta = kwargs.get("delta_theta",np.pi*2/180)
        self.perturbations = [np.array([-self.delta_x, 0, 0]), np.array([self.delta_x, 0, 0]),
                              np.array([0, -self.delta_y, 0]), np.array([0, self.delta_y, 0]),
                              np.array([0, 0, -self.delta_theta]), np.array([0, 0, self.delta_theta])]


        self.kernel_size = kwargs.get("kernel_size",5)
        self.kernel_var = kwargs.get("kernel_var",1)
        self.kernel = gaussian_kernel_2d(self.kernel_size, self.kernel_var)

        self.occ_threshold = kwargs.get("occ_threshold",0.7)

    def scan_match(self, rays, meas, initial_pose, map):
        map_res = map.res
        like_field, lower_x, lower_y = self.compute_distance_field(map, initial_pose)
        step = self.initial_step
        iterations = 0
        current_pose = initial_pose
        best_score = self.score_scan(rays, meas, current_pose, like_field, lower_x, lower_y, map_res)
        initial_score = best_score
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
        best_pose[2] = best_pose[2] % (2*np.pi)
        diff = best_score-initial_score
        return best_pose.reshape(3,1), diff

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
                q = q+like_field[cell_x, cell_y]
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

        like_field = convolve2d(binary_field, self.kernel, mode="same")

        unknown_area = prob_field == 0
        like_field[unknown_area] = 1/self.max_sensor_range

        return like_field, (lower_x - map.center_x)*map.res, (lower_y-map.center_y)*map.res

    def compute_distance_field(self, map, initial_pose):
        lower_x = np.int(np.floor((initial_pose[0] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_x
        lower_y = np.int(np.floor((initial_pose[1] - self.max_sensor_range - self.padding_dist) / map.res)) + map.center_y

        upper_x = np.int(np.floor((initial_pose[0] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_x + 1
        upper_y = np.int(np.floor((initial_pose[1] + self.max_sensor_range + self.padding_dist) / map.res)) + map.center_y + 1

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
        #unknown_area = prob_field == 0
        #dist_field[unknown_area] = self.p_unknown
        return dist_field, (lower_x - map.center_x)*map.res, (lower_y-map.center_y)*map.res

np.random.seed(0)

def test_scan_matcher():
    end = 250
    step = 5

    log = Logger()
    log.load_from_file("test_lidar22.json")

    measurements = log.get_drone_sensor_measurements("0", 0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0", 0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy, sensor_specs.sensor_attitude_bdy, num_rays=144)
    rays = sensor.ray_vectors

    scan_params =  {
        "max_iterations": 20,
        "delta_x": 0.2,
        "delta_y": 0.2,
        "step": 1,
        "sigma": 0.1,
    }
    scan = ScanMatcher(**scan_params)
    error_initial = []
    error_matched = []
    steps = []
    map = SLAM_map()
    for i in range(0,end,step):
        meas = measurements["measurements"][i]
        pose = states["states"][i][[0,1,5]]
        map.integrate_scan(pose,meas,rays)

        pose_scan = states["states"][i+step][[0,1,5]]
        meas_scan = measurements["measurements"][i]
        bounds = np.array([0.5,0.5,0.20]).reshape(3,1)
        pose_scan_pet = pose_scan + np.random.uniform(-bounds,bounds,[3,1])
        pose_matched = scan.scan_match(rays,meas_scan,pose_scan_pet,map)

        def visualize():
            plt.figure()
            plt.imshow(map.convert_grid_to_prob().transpose(),origin="lower",extent=[-map.size_x/2*map.res,map.size_x/2*map.res,-map.size_x/2*map.res,map.size_x/2*map.res])
            plt.plot(pose_scan_pet[0],pose_scan_pet[1],"ro",markersize=2)
            plt.plot(pose_scan[0],pose_scan[1],"go",markersize=2)
            plt.plot(pose_matched[0],pose_matched[1],"bo",markersize=2)
            plt.show()
        error_initial.append(pose_scan_pet-pose_scan)
        error_matched.append(pose_matched-pose_scan)
        steps.append(i)
    start = time.time()
    dist_field = scan.compute_distance_field(map,pose)
    end = time.time()
    print(end-start)
    error_initial = np.array(error_initial).squeeze()
    error_matched = np.array(error_matched).squeeze()

    f, (ax1,ax2,ax3) = plt.subplots(3,1)

    ax1.plot(steps,error_initial[:,0],"b")
    ax1.plot(steps,error_matched[:,0],"r")
    ax1.legend(["Initial","Matched"])
    print("Total initial error x:", np.sum(np.abs(error_initial[:,0])))
    print("Total mathced error x:", np.sum(np.abs(error_matched[:,0])))

    ax2.plot(steps, error_initial[:, 1], "b")
    ax2.plot(steps, error_matched[:, 1], "r")
    ax2.legend(["Initial", "Matched"])
    print("Total initial error y:", np.sum(np.abs(error_initial[:, 1])))
    print("Total mathced error y:", np.sum(np.abs(error_matched[:, 1])))

    ax3.plot(steps, error_initial[:, 2], "b")
    ax3.plot(steps, error_matched[:, 2], "r")
    ax3.legend(["Initial", "Matched"])
    print("Total initial error theta:", np.sum(np.abs(error_initial[:, 2])))
    print("Total mathced error theta:", np.sum(np.abs(error_matched[:, 2])))
    plt.show()


if __name__ == "__main__":
    test_scan_matcher()