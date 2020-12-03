import numpy as np
import multiprocessing as mp
from slam.particle import Particle
from logger.loggable import Loggable
from logger.log_entry import LogEntry
from utils.misc import compute_entropy_pose_discrete, compute_entropy_pose_gaussian
from inspect import getmodule
from copy import deepcopy

import matplotlib.pyplot as plt

class GridSLAM(Loggable):
    """

    Main Class for implementation of the GMapping SLAM Algorithm which is a particle filter based algotihm.

    Performs inizialization of the filter and the update step.

    Support for plotting and logging of the result.

    """
    def __init__(self, id, num_particles,initial_pose,rays,particle_params={}, map_params={}, scan_match_params={},
                 obs_params={}, odometry_params={},**kwargs):
        self.id = id
        self.num_particles = num_particles
        self.particle_params = particle_params
        self.particles = []
        for i in range(num_particles):
            self.particles.append(Particle(1 / num_particles, initial_pose.copy(), rays, scan_match_params=scan_match_params, map_params=map_params,
                                           odometry_params=odometry_params, obs_params=obs_params,**self.particle_params))
        self.weights = np.array([1/num_particles for i in range(num_particles)])
        self.best_particle = 0
        self.n_eff = 1/np.sum(self.weights**2)
        self.threshold_resampling = kwargs.get("threshold_resampling", 2)
        self.counter = 0
        self.entropy = 0
        self.res_pose_entropy = 0.01
        self.trajectory_entropy = 0

    def update_particles(self, measurements, odometry):
        for p in self.particles:
            p.update_particle(measurements,odometry)
        self.normalize_weights()
        if self.n_eff < self.threshold_resampling:
            print("Resampling particles!")
            self.resample_particles()
        self.compute_pose_entropy_discrete()
        print("Entropy: {:.6f}".format(self.entropy))
        self.compute_trajectory_entropy()
        print("Trajectory Entropy: {:.6f}".format(self.trajectory_entropy))
        self.counter += 1

    def update_particles_mp(self, measurements, odometry):
        process = []
        q = mp.Queue()
        for p in self.particles:
            pro = mp.Process(target=p.update_particle, args=(measurements.copy(),odometry.copy(),q))
            pro.start()
            process.append(pro)
        a = [q.get() for _ in process]
        [x.join() for x in process]
        self.particles = a
        self.normalize_weights()
        if self.n_eff < self.threshold_resampling:
            self.resample_particles()
        self.counter += 1

    def normalize_weights(self):
        for i in range(self.num_particles):
            self.weights[i] = self.particles[i].weight
        self.weights = self.weights/sum(self.weights)
        self.n_eff = 1/np.sum(self.weights**2)
        self.best_particle = np.argmax(self.weights)
        for i in range(self.num_particles):
            self.particles[i].update_weight(self.weights[i])

    def resample_particles(self):
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

    def compute_pose_entropy_discrete(self):
        poses = [p.pose for p in self.particles]
        self.entropy = compute_entropy_pose_discrete(poses,self.weights,self.res_pose_entropy)

    def compute_pose_entropy_gaussian(self):
        poses = [p.pose for p in self.particles]
        self.entropy = compute_entropy_pose_gaussian(poses,self.weights)

    def compute_trajectory_entropy(self):
        n = len(self.get_trajectory())
        entropy = 0
        for i in range(n):
            poses = [p.trajectory[i] for p in self.particles]
            entropy += compute_entropy_pose_discrete(poses,self.weights,self.res_pose_entropy)
        self.trajectory_entropy = entropy/n

    def get_best_map(self):
        return deepcopy(self.particles[self.best_particle].map)

    def get_best_particle(self):
        return deepcopy(self.particles[self.best_particle])

    def get_best_pose(self):
        return deepcopy(self.particles[self.best_particle].pose)

    def get_trajectory(self):
        return deepcopy(self.particles[self.best_particle].trajectory)

    def init_plot(self,axes):
        objects = self.particles[self.best_particle].init_plot(axes)
        return objects

    def update_plot(self, objs):
        objects = self.particles[self.best_particle].update_plot(objs)
        return objects

    def visualize_all_particles(self):
        total_map = np.zeros(self.particles[self.best_particle].map.log_prob_map.shape)
        for p in self.particles:
            total_map += p.map.log_prob_map
        prob_map = np.exp(total_map)/(1+np.exp(total_map))
        plt.figure()
        plt.imshow(prob_map.T,origin="lower")
        plt.show()

    def visualize(self):
        self.particles[self.best_particle].visualize()

    def get_time_entry(self):
        if self.counter % 10 == 0:
            return LogEntry(
                pose=self.get_best_pose(),
                map=self.get_best_map().log_prob_map.copy(),
                id=self.id,
                counter=self.counter,
                n_eff=self.n_eff,
                pose_entropy=self.entropy
            )
        else:
            return LogEntry(
                pose=self.get_best_pose(),
                id=self.id,
                counter=self.counter,
                n_eff=self.n_eff,
                pose_entropy=self.entropy
            )

    def generate_time_entry(self):
        return LogEntry(
            pose=self.get_best_pose().copy(),
            map=self.get_best_map().log_prob_map.copy(),
            id=self.id,
            counter=self.counter,
            n_eff=self.n_eff,
            pose_entropy=self.entropy
        )

    def get_info_entry(self):
        return LogEntry(
            module=getmodule(self).__name__,
            cls=type(self).__name__,
            num_particles=self.num_particles,
            id=self.id,
            map_res=self.get_best_map().res,
            map_size_x=self.get_best_map().size_x,
            map_size_y=self.get_best_map().size_y
        )