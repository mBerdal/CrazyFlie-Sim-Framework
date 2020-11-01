from slam.map import SLAM_map
from slam.scan_matcher import ScanMatcher
from slam.probability_models import OdometryModel, ObservationModel
from utils.rotation_utils import ssa
import time
import numpy as np
import matplotlib.pyplot as plt
import copy


class Particle():

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
        self.min_diff = kwargs.get("min_diff",2000.0)
        self.first = True

    def update_particle(self, measurements, odometry):
        odometry_pose = self.odometry_update(odometry)
        if not self.first:
            scan_pose, score = self.scan_matcher.scan_match(self.ray_vectors, measurements, odometry_pose, self.map)
            self.first = False
        else:
            scan_pose = odometry_pose
            score = 0

        if score < self.min_diff:
            scan_pose = odometry_pose
        #print("Odometry pose:", odometry_pose.tolist())
        #print("Scan pose:", scan_pose.tolist())
        samples = []
        for i in range(self.num_samples):
            sample_pose = scan_pose + np.random.uniform(-self.eps,self.eps,self.eps.shape)
            sample_pose[2] = sample_pose[2] % (2*np.pi)
            samples.append(sample_pose)

        mean = np.zeros([3,1])
        normalizer = 0
        n_obs = 0
        n_mot = 0

        self.observation_model.compute_likelihood_field_dist(scan_pose,self.map)

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
        pose[2] = pose[2] % (2*np.pi)
        self.pose = pose
        self.map.integrate_scan(pose, measurements, self.ray_vectors)
        #print("Motion:",n_mot)
        #print("Observation:",n_obs)

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

    def __deepcopy__(self, memodict={},**kwargs):
        par = Particle(copy.deepcopy(self.weight),copy.deepcopy(self.pose),self.ray_vectors,
                       scan_match_params=self.scan_match_params,map_params=self.map_params,
                       odometry_params=self.odometry_params,obs_params=self.obs_params,**kwargs)
        par.map = self.map.__deepcopy__()
        return par

    def init_plot(self, axis):
        im = self.map.init_plot(axis)
        d = plt.Circle((self.pose[0], self.pose[1]), radius=0.1, color="red")
        axis.add_patch(d)
        return {"drone": d, "map": im}

    def update_plot(self,objects):
        objects["map"] = self.map.update_plot(objects["map"])
        objects["drone"].set_center((self.pose[0], self.pose[1]))
        return objects

    def visualize(self):
        plt.figure()
        plt.imshow(self.map.convert_grid_to_prob().transpose(),"Greys",origin="lower",
                   extent=[-self.map.size_x/2*self.map.res,self.map.size_x/2*self.map.res,-self.map.size_y/2*self.map.res,self.map.size_y/2*self.map.res])
        plt.plot(self.pose[0], self.pose[1], "o", color="red",markersize=2)
        plt.show()


def diff_covariance(sample,mean):
    diff = np.zeros([3,1])
    diff[0:2] = sample[0:2]-mean[0:2]
    diff[2] = ssa(sample[2],mean[2])
    return diff

