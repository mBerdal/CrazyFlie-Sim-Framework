from slam.map import SLAM_map
from slam.scan_matcher import ScanMatcher
from slam.probability_models import OdometryModel, ObservationModel
from utils.rotation_utils import ssa
import numpy as np
import matplotlib.pyplot as plt
import copy
from slam.icp import icp


class Particle:

    def __init__(self, initial_weight, initial_pose, ray_vectors, scan_match_params={}, map_params={}, odometry_params={}, obs_params={},**kwargs):
        self.weight = initial_weight
        self.pose = initial_pose
        self.ray_vectors = ray_vectors

        self.map = SLAM_map(**map_params)
        self.scan_matcher = ScanMatcher(**scan_match_params)
        self.motion_model = OdometryModel(**odometry_params)
        self.observation_model = ObservationModel(self.ray_vectors, **obs_params)

        self.num_samples = kwargs.get("num_samples", 40)
        self.eps = kwargs.get("eps",np.array([0.1,0.1,0.05]).reshape(3,1))

        self.map_params = map_params
        self.scan_match_params = scan_match_params
        self.odometry_params = odometry_params
        self.obs_params = obs_params
        self.min_diff = kwargs.get("min_diff",2.0)
        self.translation_sample = np.linspace(-0.05,0.05,5)
        self.angular_sample = np.linspace(-np.deg2rad(3),np.deg2rad(3),5)
        self.trajectory = [initial_pose.copy()]
        self.counter = 0
        self.previous_points = None

    def update_particle(self, measurements, odometry):
        odometry_pose = self.odometry_update(odometry)
        #if self.previous_points is not None:
        #    points = self.ray_vectors[0:2,:]*measurements + odometry_pose[0:2]
        #    points = points[~np.any(np.isinf(points), axis=1)]
        #    translation = icp(self.previous_points.T,points.T)
        scan_pose, score = self.scan_matcher.scan_match(self.ray_vectors, measurements, odometry_pose, self.map)
        samples = []
        for x in self.translation_sample:
            for y in self.translation_sample:
                for a in self.angular_sample:
                    sample_pose = scan_pose + np.array([x,y,a]).reshape(3,1)
                    sample_pose[2] = sample_pose[2] % (2 * np.pi)
                    samples.append(sample_pose)
        mean = np.zeros([3,1])
        normalizer = 0
        n_obs = 0
        n_mot = 0

        self.observation_model.compute_likelihood_field(scan_pose,self.map)

        for s in samples:
            l_obs = self.observation_model.likelihood(s, measurements)
            l_mot = self.motion_model.likelihood(s, self.pose, odometry)
            likelihood = l_obs*l_mot
            mean[0:2] += s[0:2]*likelihood
            mean[2] += ssa(s[2],0)*likelihood
            normalizer += likelihood
            n_obs += l_obs
            n_mot += l_mot

        try:
            mean = mean/normalizer
        except:
            mean = mean/(normalizer+1e-200)
        covariance = np.zeros([3,3])

        for s in samples:
            likelihood = self.observation_model.likelihood(s, measurements) * self.motion_model.likelihood(
                s, self.pose, odometry)
            diff = diff_covariance(s,mean)
            covariance += diff*diff.transpose()*likelihood
        try:
            covariance = covariance/(normalizer)
        except:
            covariance = covariance/(normalizer+1e-200)

        try:
            pose = np.random.multivariate_normal(mean.squeeze(),covariance).reshape(3,1)
            self.weight = self.weight * normalizer
        except:
            pose = scan_pose
            self.weight = 1e-200

        if self.counter < 3:
            self.counter += 1
            pose = self.pose

        pose[2] = pose[2] % (2*np.pi)

        self.pose = pose
        self.map.integrate_scan(pose, measurements, self.ray_vectors)
        self.trajectory.append(self.pose.copy())
        self.previous_points = self.ray_vectors[0:2,:] * measurements + self.pose[0:2]
        #q.put(self)

    def update_weight(self, weight):
        self.weight = weight

    def odometry_update(self, odometry):
        diff = np.zeros([3,1])
        diff[0:2] = (odometry[0, 0:2] - odometry[1, 0:2]).reshape(2,1)
        diff[2] = ssa(odometry[0,2],odometry[1,2])
        pose = np.zeros([3,1])

        pose[0] = self.pose[0] + diff[0]*np.cos(self.pose[2]) - diff[1]*np.sin(self.pose[2])
        pose[1] = self.pose[1] + diff[0]*np.sin(self.pose[2]) + diff[1]*np.cos(self.pose[2])
        pose[2] = (self.pose[2] + diff[2]) % (2*np.pi)
        return pose

    def __deepcopy__(self, memodict={}, **kwargs):
        par = Particle(copy.deepcopy(self.weight),copy.deepcopy(self.pose),self.ray_vectors,
                       scan_match_params=self.scan_match_params,map_params=self.map_params,
                       odometry_params=self.odometry_params,obs_params=self.obs_params,**kwargs)
        par.map = self.map.__deepcopy__()
        par.trajectory = self.trajectory.copy()
        par.counter = self.counter
        return par

    def init_plot(self, axis):
        im = self.map.init_plot(axis)
        d = plt.Circle((self.pose[0], self.pose[1]), radius=0.1, color="red")
        axis.add_patch(d)
        t,  = axis.plot([t[0] for t in self.trajectory], [t[1] for t in self.trajectory], "-o", color="green", markersize=1)
        return {"drone": d, "map": im, "trajectory": t}

    def update_plot(self,objects):
        objects["map"] = self.map.update_plot(objects["map"])
        objects["drone"].set_center((self.pose[0], self.pose[1]))
        objects["trajectory"].set_xdata([t[0] for t in self.trajectory])
        objects["trajectory"].set_ydata([t[1] for t in self.trajectory])
        return objects

    def visualize(self):
        plt.figure()
        plt.imshow(self.map.convert_grid_to_prob().transpose(),"Greys",origin="lower",
                   extent=[-self.map.size_x/2*self.map.res,self.map.size_x/2*self.map.res,-self.map.size_y/2*self.map.res,self.map.size_y/2*self.map.res])
        plt.plot(self.pose[0], self.pose[1], "o", color="red",markersize=2)
        plt.plot([t[0] for t in self.trajectory], [t[1] for t in self.trajectory], "-o", color="blue", markersize=2)
        plt.pause(0.1)


def diff_covariance(sample,mean):
    diff = np.zeros([3,1])
    diff[0:2] = sample[0:2]-mean[0:2]
    diff[2] = ssa(sample[2],mean[2])
    return diff

