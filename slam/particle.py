from slam.map import SLAM_map
from slam.scan_matcher import ScanMatcher
from slam.probability_models import OdometryModel, ObservationModel
from utils.rotation_utils import ssa
import time
import numpy as np
import copy

class Particle():

    def __init__(self, initial_weight, initial_pose, ray_vectors, scan_match_params={}, map_params={}, odometry_params={}, obs_params={}):
        self.map = SLAM_map(**map_params)
        self.weight = initial_weight
        self.pose = initial_pose
        self.K = 20
        self.ray_vectors = ray_vectors
        self.scan_matcher = ScanMatcher(**scan_match_params)
        self.motion_model = OdometryModel(**odometry_params)
        self.observation_model = ObservationModel(self.ray_vectors, **obs_params)

        self.eps = np.array([0.1,0.1,0.05]).reshape(3,1)

        self.map_params = map_params
        self.scan_match_params = scan_match_params
        self.odometry_params = odometry_params
        self.obs_params = obs_params

    def update_particle(self, measurements, odometry):
        odometry_pose = self.odometry_update(odometry[0,:]-odometry[1,:])
        scan_pose = self.scan_matcher.scan_match(self.ray_vectors, measurements, odometry_pose, self.map)

        samples = []
        for i in range(self.K):
            sample_pose = scan_pose + np.random.uniform(-self.eps,self.eps,self.eps.shape)
            sample_pose[2] = sample_pose[2] % (2*np.pi)
            samples.append(sample_pose)

        mean = np.zeros([3,1])
        normalizer = 0

        self.observation_model.compute_likelihood_field(self.map)

        for s in samples:
            likelihood = self.observation_model.likelihood(self.map, s, measurements)*self.motion_model.likelihood(s, self.pose, odometry)
            mean[0:2] += s[0:2]*likelihood
            mean[2] += ssa(s[2],0)*likelihood
            normalizer += likelihood

        mean = mean/normalizer
        covariance = np.zeros([3,3])

        for s in samples:
            likelihood = self.observation_model.likelihood(self.map, s, measurements) * self.motion_model.likelihood(
                s, self.pose, odometry)
            diff = diff_covariance(s,mean)
            covariance += diff*diff.transpose()*likelihood

        covariance = covariance/(normalizer)
        pose = np.random.multivariate_normal(mean.squeeze(),covariance).reshape(3,1)
        pose[2] = pose[2] % (2*np.pi)
        self.pose = pose
        self.weight = self.weight*normalizer
        self.map.integrate_scan(pose.copy(), measurements, self.ray_vectors)

    def update_weight(self, weight):
        self.weight = weight

    def odometry_update(self, odometry):
        pose = np.zeros([3,1])
        pose[0] = self.pose[0] + odometry[0]*np.cos(odometry[2]) - odometry[1]*np.sin(odometry[2])
        pose[1] = self.pose[1] + odometry[0]*np.sin(odometry[2]) + odometry[1]*np.cos(odometry[2])
        pose[2] = (self.pose[2] + odometry[2]) % (2*np.pi)
        return pose

    def __deepcopy__(self, memodict={}):
        par = Particle(copy.deepcopy(self.weight),copy.deepcopy(self.pose),self.ray_vectors,
                       scan_match_params=self.scan_match_params,map_params=self.map_params,
                       odometry_params=self.odometry_params,obs_params=self.obs_params)
        par.map = self.map.__deepcopy__()
        return par

def diff_covariance(sample,mean):
    diff = np.zeros([3,1])
    diff[0:2] = sample[0:2]-mean[0:2]
    diff[2] = ssa(sample[2],mean[2])
    return diff

