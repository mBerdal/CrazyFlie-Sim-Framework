import numpy as np
from slam.particle import Particle
from logger.loggable import Loggable
from logger.log_entry import LogEntry
from utils.misc import compute_entropy_pose_discrete, compute_entropy_pose_gaussian
from inspect import getmodule
from copy import deepcopy
from utils.rotation_utils import ssa
from slam.scan_matcher import ScanMatcher
from slam.probability_models import OdometryModel, ObservationModel
from matplotlib import cm

import matplotlib.pyplot as plt

class GridSLAM(Loggable):
    """
    Main Class for implementation of the GMapping SLAM Algorithm which is a particle filter based algorithm.

    Updates the particle filter based on the given LIDAR observations and odometry reading.
    """

    def __init__(self, id, num_particles,initial_pose,rays, particle_params={}, map_params={}, scan_match_params={},
                 obs_params={}, odometry_params={},**kwargs):
        self.id = id
        self.num_particles = num_particles
        self.particle_params = particle_params

        self.particles = []
        for i in range(num_particles):
            self.particles.append(Particle(i, 1 / num_particles, deepcopy(initial_pose), map_params=map_params))
        self.weights = np.array([1/num_particles for _ in range(num_particles)])
        self.best_particle = 0

        self.n_eff = 1/np.sum(self.weights**2)
        self.threshold_resampling = kwargs.get("threshold_resampling", 0.5)

        self.counter = 0
        self.entropy = 0
        self.res_pose_entropy = 0.1
        self.trajectory_entropy = 0

        self.scan_matcher = ScanMatcher(**scan_match_params)
        self.motion_model = OdometryModel(**odometry_params)
        self.observation_model = ObservationModel(rays, **obs_params)

        self.rays = rays

        self.translation_sample = np.linspace(-0.025, 0.025, 5)
        self.angular_sample = np.linspace(-0.025, 0.025, 5)

        self.resampling_counter = 0

        self.max_diff = 0.3

    def update_particles(self, range_sensor, odometry):
        #Update all particles
        for p in self.particles:
            #Compute initial estimate based on odometry and improve the estimate using scan matching.
            odometry_pose = self.odometry_update(p.get_pose(), odometry)
            scan_pose, score = self.scan_matcher.scan_match(self.rays, range_sensor, odometry_pose, p.map)
            if np.linalg.norm(scan_pose[0:2] - odometry_pose[0:2]) > self.max_diff:
                scan_pose[0:2] = odometry_pose[0:2]

            #Sample around the scan matched estimate.
            samples = []
            for x in self.translation_sample:
                for y in self.translation_sample:
                    for a in self.angular_sample:
                        sample_pose = scan_pose + np.array([x, y, a]).reshape(3, 1)
                        sample_pose[2] = sample_pose[2] % (2 * np.pi)
                        samples.append(sample_pose)

            mean = np.zeros([3, 1])

            normalizer = 0.0
            n_obs = 0.0
            n_mot = 0.0

            self.observation_model.compute_likelihood_field(scan_pose, p.map)
            #Compute the mean of the samples based on the observation and motion likelihood.
            likelihoods = []
            for s in samples:
                l_obs = self.observation_model.likelihood(s, range_sensor)
                l_mot = self.motion_model.likelihood(s.copy(), p.pose.copy(), odometry)
                likelihood = l_obs*l_mot
                likelihoods.append(likelihood)
                mean[0:2] += s[0:2] * likelihood
                mean[2] += ssa(s[2], 0) * likelihood
                normalizer += likelihood
                n_obs += l_obs
                n_mot += l_mot

            mean = mean / normalizer
            covariance = np.zeros([3, 3])

            #Compute the covariance matrix
            for (i, s) in enumerate(samples):
                diff = diff_covariance(s, mean)
                covariance += diff * diff.transpose() * likelihoods[i]

            covariance = covariance / normalizer

            #Sample from the improved proposal distribution
            try:
                pose = np.random.multivariate_normal(mean.squeeze(), covariance).reshape(3, 1)
                p.weight = p.weight * normalizer
            except:
                pose = scan_pose
                self.weight = 1e-200

            #In the initialization phase the pose is not updated, only the map
            if self.counter < 3:
                pose = p.get_pose()

            pose[2] = pose[2] % (2 * np.pi)
            p.pose = deepcopy(pose)
            p.map.integrate_scan(p.pose, range_sensor, self.rays)
            p.trajectory.append(p.pose.copy())
            p.update_loop_graph()

        #Normalize the particles and performe resampling if neccesary
        self.normalize_weights()
        if self.n_eff < self.threshold_resampling:
            self.resampling_counter += 1
            print("Resampling particles! Resampling num:", self.resampling_counter )
            self.resample_particles()
        self.compute_pose_entropy_discrete()
        print("Entropy: {:.6f}".format(self.entropy))
        self.compute_trajectory_entropy()
        print("Trajectory Entropy: {:.6f}".format(self.trajectory_entropy))
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
            ind = np.argmax(cum_weights > r)
            new_p = self.particles[ind].__deepcopy__(**self.particle_params)
            new_particles.append(new_p)
        self.particles = new_particles
        for p in self.particles:
            p.update_weight(1/self.num_particles)
        self.weights[:] = 1/self.num_particles

    def odometry_update(self, pose, odometry):
        diff = np.zeros([3, 1])
        diff[0:2] = (odometry[0, 0:2] - odometry[1, 0:2]).reshape(2, 1)
        diff[2] = ssa(odometry[0, 2], odometry[1, 2])
        pose_odo = np.zeros([3, 1])

        pose_odo[0] = pose[0] + diff[0] * np.cos(pose[2]) - diff[1] * np.sin(pose[2])
        pose_odo[1] = pose[1] + diff[0] * np.sin(pose[2]) + diff[1] * np.cos(pose[2])
        pose_odo[2] = (pose[2] + diff[2]) % (2 * np.pi)
        return pose_odo

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

    def get_loop_graph(self):
        return deepcopy(self.particles[self.best_particle].loop_graph), deepcopy(self.particles[self.best_particle].current_node)

    def get_loop_graph_log(self):
        return self.particles[self.best_particle].loop_graph_log()

    def init_plot(self,axes):
        objects = self.particles[self.best_particle].init_plot(axes)
        return objects

    def update_plot(self, objs):
        objects = self.particles[self.best_particle].update_plot(objs)
        return objects

    def visualize(self):
        self.particles[self.best_particle].visualize()

    def visualize_loop_graph(self):
        self.particles[self.best_particle].visualize_loop_graph()

    def get_time_entry(self):
        if self.counter % 8 == 0:
            return LogEntry(
                pose=self.get_best_pose(),
                map=self.get_best_map().log_prob_map.copy(),
                id=self.id,
                counter=self.counter,
                n_eff=self.n_eff,
                pose_entropy=self.entropy,
                loop_graph=self.get_loop_graph_log()
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

    def test_likelihood(self, pose, measurements):
        n = 101
        pet = 0.20
        x_grid = np.linspace(-pet, pet, n)
        y_grid = np.linspace(-pet, pet, n)
        res = np.zeros([n,n])
        for r in range(n):
            for c in range(n):
                p = np.zeros([3,1])
                p[0] = pose[0] + x_grid[c]
                p[1] = pose[1] + y_grid[r]
                p[2] = pose[2]
                res[r,c] = self.observation_model.likelihood(p, measurements)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X, Y = np.meshgrid(x_grid, y_grid)

        # Plot the surface.
        ax.plot_surface(X, Y, res, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()


def diff_covariance(sample,mean):
    diff = np.zeros([3,1])
    diff[0:2] = sample[0:2]-mean[0:2]
    diff[2] = ssa(sample[2],mean[2])
    return diff