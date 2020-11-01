from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve2d
from slam.map import SLAM_map
from utils.rotation_utils import ssa, rot_matrix_2d
from utils.misc import gaussian_kernel_2d
from logger.logger import Logger
import matplotlib.pyplot as plt
import matplotlib
from sensor.lidar_sensor import LidarSensor
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
    def __init__(self, ray_vectors,**kwargs):
        self.kernel_size = kwargs.get("kernel_size",5)
        self.kernel_var = kwargs.get("kernel_var",1)
        self.kernel = gaussian_kernel_2d(self.kernel_size,self.kernel_var)
        self.ray_vectors = ray_vectors
        self.occ_threshold = kwargs.get("occ_threshold",0.7)
        self.eps = 0.1

        self.like_field = None
        self.lower_x = 0
        self.lower_y = 0
        self.map_res = None

        self.sigma = kwargs.get("sigma",0.5)
        self.p_unknown = 0.2

        self.max_sensor_range = kwargs.get("max_sensor_range", 4)
        self.padding_dist = 2.0

    def likelihood(self, pose, measurements):
        q = 1
        pose = pose.squeeze()
        rays = rot_matrix_2d(pose[2]) @ self.ray_vectors[0:2, :]
        for ind, meas in enumerate(measurements):
            if meas == np.inf:
                continue
            else:
                end_point = pose[0:2] + rays[:, ind] * meas
            cell_x = np.int(np.floor((end_point[0] - self.lower_x) / self.map_res))
            cell_y = np.int(np.floor((end_point[1] - self.lower_y) / self.map_res))
            try:
                q = q*(self.like_field[cell_x, cell_y] + self.eps)
            except:
                pass
        return q

    def compute_likelihood_field(self, map):
        prob_field = map.log_prob_map.copy()
        binary_field = prob_field > np.log(self.occ_threshold/(1-self.occ_threshold))
        like_field = convolve2d(binary_field, self.kernel, mode="same")
        unknown_area = prob_field == 0
        like_field[unknown_area] = 1/self.max_sensor_range**2
        self.like_field = like_field

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
        unknown_area = prob_field == 0
        dist_field[unknown_area] = self.p_unknown
        self.like_field = dist_field
        self.lower_x = (lower_x - map.center_x) * map.res
        self.lower_y = (lower_y - map.center_y) * map.res
        self.map_res = map.res

def visualize_observations(pose, rays, meas, map, likelihood):
    rays = rot_matrix_2d(pose[2].squeeze()) @ rays[0:2,:]
    fig, ax = plt.subplots(1)
    ax.imshow(likelihood.transpose(), origin="lower")
    drone = plt.Circle([pose[0], pose[1]],radius=3)
    drone_coord_x= pose[0]/map.res + map.center_x
    drone_coord_y= pose[1]/map.res + map.center_y
    ax.add_patch(drone)
    for i in range(len(meas)):
        ray = rays[:, i]
        if meas[i] == np.inf:
            continue
        else:
            end_point = pose[0:2] + ray.reshape(2, 1) * meas[i]
            end_point_x = end_point[0] / map.res + map.center_x
            end_point_y = end_point[1] / map.res + map.center_y
        ax.plot([drone_coord_x,end_point_x],[drone_coord_y,end_point_y],color="red")
    plt.pause(1)
    plt.close()

def test_prob_motion():
    log = Logger()
    log.load_from_file("test_lidar22.json")

    measurements = log.get_drone_sensor_measurements("0", 0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0", 0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy, sensor_specs.sensor_attitude_bdy, num_rays=144)
    rays = sensor.ray_vectors

    map = SLAM_map()
    model = OdometryModel()
    for i in range(10,250,10):
        pose1 = states["states"][i-10][[0,1,5]]
        pose2 = states["states"][i][[0,1,5]]
        print(pose2-pose1)

        odometry = np.concatenate([pose1,pose2],axis=1).transpose()



        x = np.arange(pose2[0] - 1, pose2[0] + 1 + 0.02, 0.02)
        y = np.arange(pose2[1] - 1, pose2[1] + 1 + 0.02, 0.02)
        xx, yy = np.meshgrid(x, y)
        likelihood = np.zeros(xx.shape)
        for x in range(xx.shape[0]):
            for y in range(xx.shape[1]):
                likelihood[x,y] = model.likelihood(np.array([xx[x,y],yy[x,y],pose2[2]],np.float).reshape(3,1),pose1,odometry)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, likelihood, cmap=matplotlib.cm.coolwarm)
        plt.show()

    rots = np.arange(-np.pi,np.pi,np.pi/180)
    likelihood = np.zeros(rots.shape)
    for i in range(len(rots)):
        p = np.zeros([3,1])
        p[0:2] = pose2[0:2]
        p[2] = rots[i]
        likelihood[i] = model.likelihood(p,pose1,odometry)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(rots,likelihood)
    plt.show()


def test_prob_obs():
    log = Logger()
    log.load_from_file("test_lidar22.json")

    measurements = log.get_drone_sensor_measurements("0", 0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0", 0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy, sensor_specs.sensor_attitude_bdy, num_rays=144)
    rays = sensor.ray_vectors

    map = SLAM_map()

    end = 250
    step = 10
    obs = ObservationModel(rays, kernel_size=5, kernel_variance=1)
    model = OdometryModel()
    obs.compute_likelihood_field(map)

    def visualize_map(obs_model):
        plt.figure()
        plt.imshow(obs.like_field.transpose(),origin="lower")
        plt.colorbar()
        plt.show()
    #visualize_map(obs)
    for i in range(step,end,step):
        meas = measurements["measurements"][i-step]
        pose = states["states"][i-step][[0,1,5]]
        map.integrate_scan(pose,meas,rays)
        x = np.arange(pose[0] - 1, pose[0] + 1 + 0.01, 0.01)
        y = np.arange(pose[1] - 1, pose[1] + 1 + 0.01, 0.01)
        xx, yy = np.meshgrid(x, y)
        rot = np.zeros(xx.shape)

        pose1 = states["states"][i - step][[0, 1, 5]]
        pose2 = states["states"][i][[0, 1, 5]]

        odometry = np.concatenate([pose1, pose2], axis=1).transpose()


        meas1 = measurements["measurements"][i]
        pose1 = states["states"][i][[0,1,5]]
        x = np.arange(pose1[0]-1,pose1[0]+1+0.02,0.02)
        y = np.arange(pose1[1]-1,pose1[1]+1+0.02,0.02)
        xx, yy = np.meshgrid(x,y)
        likelihood = np.zeros(xx.shape)
        rot = np.zeros(xx.shape)
        obs.compute_likelihood_field_dist(pose1,map)
        visualize_map(obs)
        for x in range(xx.shape[0]):
            for y in range(xx.shape[1]):
                likelihood[x,y] = obs.likelihood(map,np.array([xx[x,y],yy[x,y],pose1[2]],np.float).reshape(3,1),meas1)*model.likelihood(np.array([xx[x,y],yy[x,y],pose1[2]],np.float).reshape(3,1),pose,odometry)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xx, yy, likelihood, cmap=matplotlib.cm.coolwarm)
        plt.pause(0.1)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx,yy,likelihood,cmap=matplotlib.cm.coolwarm)
    plt.show()

    rots = np.arange(-np.pi, np.pi, np.pi / 180)
    likelihood = np.zeros(rots.shape)
    for i in range(len(rots)):
        pose[2] = rots[i]
        likelihood[i] = obs.likelihood(map,pose,meas)

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(rots, likelihood)
    plt.show()

if __name__ == "__main__":
    test_prob_motion()

