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
        self.alpha1 = kwargs.get("alpha1", 1)
        self.alpha2 = kwargs.get("alpha2", 1)
        self.alpha3 = kwargs.get("alpha3", 1)
        self.alpha4 = kwargs.get("alpha4", 1)

    def likelihood(self,new_pose, prev_pose, u):
        od_prev = u[0,:]
        od_new = u[1,:]

        delta_rot1_od = ssa(np.arctan2(od_new[1]-od_prev[1],od_new[0]-od_new[0]),od_prev[2])
        delta_trans_od = np.linalg.norm(od_prev[0:1]-od_new[0:1])
        delta_rot2_od = ssa(od_new[2],(od_prev[2]-delta_rot1_od)%(2*np.pi))

        delta_rot1_pose = ssa(np.arctan2(new_pose[1] - prev_pose[1], new_pose[0] - prev_pose[0]),prev_pose[2])
        delta_trans_pose = np.linalg.norm(prev_pose[0:1] - new_pose[0:1])
        delta_rot2_pose = ssa(new_pose[2], (prev_pose[2] - delta_rot1_pose)%(2*np.pi))

        std1 = self.alpha1*delta_rot1_od**2+ self.alpha2*delta_trans_od**2
        err1 = ssa(delta_rot1_pose,delta_rot1_od)
        p1 = norm.pdf(err1,scale=std1)

        std2 = self.alpha3*delta_trans_od**2+self.alpha4*(delta_rot1_od**2 + delta_rot2_od**2)
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
        self.z_random = 0.01
        self.z_max = 0.4
        self.like_field = None
        self.max_range = kwargs.get("max_range",4)

    def likelihood(self, map, pose, measurements):
        q = 1
        pose = pose.squeeze()
        rays = rot_matrix_2d(pose[2]) @ self.ray_vectors[0:2, :]
        for ind, meas in enumerate(measurements):
            if meas == np.inf:
                continue
            else:
                end_point = pose[0:2] + rays[:, ind] * meas
            cell = map.world_coordinate_to_grid_cell(end_point)
            try:
                q = q*(self.like_field[cell[0], cell[1]] + self.z_random/self.z_max)
            except:
                pass
        return q

    def compute_likelihood_field(self, map):
        prob_field = map.log_prob_map.copy()
        binary_field = prob_field > np.log(self.occ_threshold/(1-self.occ_threshold))
        like_field = convolve2d(binary_field, self.kernel, mode="same")
        unknown_area = prob_field == 0
        like_field[unknown_area] = 1/self.max_range**2
        self.like_field = like_field


def test_prob_motion():
    log = Logger()
    log.load_from_file("test_lidar.json")

    measurements = log.get_drone_sensor_measurements("0", 0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0", 0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy, sensor_specs.sensor_attitude_bdy, num_rays=144)
    rays = sensor.ray_vectors

    map = SLAM_map()

    pose1 = states["states"][0][[0,1,5]]
    pose2 = states["states"][10][[0,1,5]]

    odometry = np.concatenate([pose1,pose2],axis=1).transpose()

    model = OdometryModel()

    x = np.arange(pose2[0] - 1, pose2[0] + 1 + 0.01, 0.01)
    y = np.arange(pose2[1] - 1, pose2[1] + 1 + 0.01, 0.01)
    xx, yy = np.meshgrid(x, y)
    rot = np.zeros(xx.shape)
    likelihood = np.zeros(xx.shape)
    for x in range(xx.shape[0]):
        for y in range(xx.shape[1]):
            likelihood[x,y] = model.likelihood(np.array([xx[x,y],yy[x,y],0]).reshape(3,1),pose1,odometry)

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


def test_prob():
    log = Logger()
    log.load_from_file("test_lidar.json")

    measurements = log.get_drone_sensor_measurements("0", 0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0", 0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy, sensor_specs.sensor_attitude_bdy, num_rays=72)
    rays = sensor.ray_vectors

    map = SLAM_map()

    for i in range(100):
        meas = measurements["measurements"][i]
        pose = states["states"][i][[0,1,5]]
        map.integrate_scan(pose,meas,rays)

    obs = ObservationModel(rays,kernel_size=7,kernel_variance=0.5)
    obs.compute_likelihood_field(map)



    def visualize_map(obs_model):
        plt.figure()
        plt.imshow(obs.like_field.transpose(),origin="lower")
        plt.colorbar()
        plt.show()
    visualize_map(obs)

    x = np.arange(pose[0]-1,pose[0]+1+0.01,0.01)
    y = np.arange(pose[1]-1,pose[1]+1+0.01,0.01)
    xx, yy = np.meshgrid(x,y)
    rot = np.zeros(xx.shape)

    def visualize_observations(pose, rays, meas, map, likelihood):
        rays = rot_matrix_2d(pose[2].squeeze()) @ rays[0:2,:]
        fig, ax = plt.subplots(1)
        ax.imshow(likelihood.transpose(), origin="lower")
        drone = plt.Circle([pose[0],pose[1]],radius=3)
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

    likelihood = np.zeros(xx.shape)

    for x in range(xx.shape[0]):
        for y in range(xx.shape[1]):
            likelihood[x,y] = obs.likelihood(map,np.array([xx[x,y],yy[x,y],0]).reshape(3,1),meas)
            #visualize_observations(np.array([xx[x,y],yy[x,y],0]).reshape(3,1),rays,meas,map,obs.like_field)

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
    pass

