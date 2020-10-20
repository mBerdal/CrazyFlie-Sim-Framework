import numpy as np
from slam.particle import Particle
import copy
import time
class Grid_SLAM():

    def __init__(self, num_particles,initial_pose,rays,map_params={},scan_match_params={},obs_params={}, odometry_params={}):
        self.num_particles = num_particles
        self.particles = []
        for i in range(num_particles):
            self.particles.append(Particle(1 / num_particles, initial_pose.copy(), rays, scan_match_params=scan_match_params, map_params=map_params,
                                           odometry_params=odometry_params, obs_params=obs_params))
        self.weights = np.array([1/num_particles for i in range(num_particles)])
        self.best_particle = 0
        self.n_eff = 1/np.sum(self.weights**2)
        self.T = self.num_particles/2


    def update_particles(self, measurements, odometry):
        for p in self.particles:
            p.update_particle(measurements,odometry)
        self.normalize_weights()
        if self.n_eff < self.T:
            self.resample_particles()

    def normalize_weights(self):
        for i in range(self.num_particles):
            self.weights[i] = self.particles[i].weight
        self.weights = self.weights/sum(self.weights)
        self.n_eff = 1/np.sum(self.weights**2)
        self.best_particle = np.argmax(self.weights)
        for i in range(self.num_particles):
            self.particles[i].update_weight(self.weights[i])

    def resample_particles(self):
        print("Resampling particles")
        cum_weights = np.cumsum(self.weights)
        new_particles = []
        for i in range(self.num_particles):
            r = np.random.uniform(0,1)
            ind = np.argmax(cum_weights>r)
            new_p = copy.deepcopy(self.particles[ind])
            new_particles.append(new_p)
        self.particles = new_particles
        for p in self.particles:
            p.update_weight(1/self.num_particles)
        self.weights[:] = 1/self.num_particles
