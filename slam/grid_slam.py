import numpy as np
from slam.particle import Particle
import matplotlib.pyplot as plt
import copy
import time
class Grid_SLAM():

    def __init__(self, num_particles,initial_pose,rays,particle_params={},map_params={},scan_match_params={},obs_params={}, odometry_params={},**kwargs):
        self.num_particles = num_particles
        self.particle_params = particle_params
        self.particles = []
        for i in range(num_particles):
            self.particles.append(Particle(1 / num_particles, initial_pose.copy(), rays, scan_match_params=scan_match_params, map_params=map_params,
                                           odometry_params=odometry_params, obs_params=obs_params,**self.particle_params))
        self.weights = np.array([1/num_particles for i in range(num_particles)])
        self.best_particle = 0
        self.n_eff = 1/np.sum(self.weights**2)
        self.threshold_resampling = kwargs.get("threshold_resampling", self.num_particles / 2)


    def update_particles(self, measurements, odometry):
        for p in self.particles:
            p.update_particle(measurements,odometry)
        self.normalize_weights()
        print(self.n_eff)
        if self.n_eff < self.threshold_resampling:
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
            new_p = self.particles[ind].__deepcopy__(**self.particle_params)
            new_particles.append(new_p)
        self.particles = new_particles
        for p in self.particles:
            p.update_weight(1/self.num_particles)
        self.weights[:] = 1/self.num_particles

    def get_best_map(self):
        return self.particles[self.best_particle].map

    def get_best_particle(self):
        return self.particles[self.best_particle]

    def init_plot(self,axes):
        objects = self.particles[self.best_particle].init_plot(axes)
        return objects

    def update_plot(self, objs):
        objects = self.particles[self.best_particle].update_plot(objs)
        return objects

