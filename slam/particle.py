from slam.map import SLAM_map
import numpy as np

class Particle():

    def __init__(self, initial_weight, map_params, ray_vectors):
        self.map = SLAM_map(map_params)
        self.weight = initial_weight
        self.pose = np.zeros(3)
        self.K = 10
        self.ray_vectors = ray_vectors

    def update_particle(self, measurments, odometry):
        odometry_pose = self.odometry_update(odometry)
        scan_pose = self.scan_match_pose(odometry_pose,self.map,measurments)

        if scan_pose == None:
            #draw x from odometry distribution
            #update weight based on p(z|m,x)
            pass
        else:
            samples = []
            for i in range(self.K):
                #Sample around the scan matched pose
                pass
            mean = np.zeros(3)
            normalizer = 0
            for x in samples:
                #update mean based on probability of observation and odometry
                #update normalizer based on the probability of observation and odometry
                pass
            mean = mean/normalizer
            covariance = np.zeros(3,3)
            for x in samples:
                #update covariance based on the samples and computed mean
                pass
            covariance = covariance/normalizer
            self.pose = None #Sample from the distribution given by mean and covariance
            self.weight = self.weight*normalizer

        self.map.integrate_scan(self.pose,measurments,self.ray_vectors)
        pass

    def update_weight(self, weight):
        self.weight = weight

    def odometry_update(self, odometry):
        self.pose[0] = self.pose[0] + odometry[0]*np.cos(odometry[2]) - odometry[1]*np.sin(odometry[2])
        self.pose[1] = self.pose[1] + odometry[0]*np.sin(odometry[2]) + odometry[1]*np.cos(odometry[2])
        self.pose[2] = self.pose[2] + odometry[2]

    def scan_match_pose(self, initial_pose, map, measurments):
        pass

    def sample_around_proposal(self):
        pass

    def compute_gaussian_proposal(self):
        pass