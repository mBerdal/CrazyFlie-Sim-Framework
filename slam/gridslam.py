import numpy as np
import multiprocessing as mp
from slam.particle import Particle
from logger.loggable import Loggable
from logger.log_entry import LogEntry
from utils.misc import compute_entropy_pose_discrete, compute_entropy_pose_gaussian
from inspect import getmodule
from copy import deepcopy
from utils.rotation_utils import ssa, rot_matrix_2d
from slam.scan_matcher import ScanMatcher
from slam.probability_models import OdometryModel, ObservationModel
from math import floor
import time
from matplotlib import cm

import matplotlib.pyplot as plt

class GridSLAM(Loggable):
    """

    Main Class for implementation of the GMapping SLAM Algorithm which is a particle filter based algotihm.

    Performs inizialization of the filter and the update step.

    Support for plotting and logging of the result.

    """
    def __init__(self, id, num_particles,initial_pose,rays, particle_params={}, map_params={}, scan_match_params={},
                 obs_params={}, odometry_params={},**kwargs):
        self.id = id
        self.num_particles = num_particles
        self.particle_params = particle_params
        self.particles = []
        for i in range(num_particles):
            self.particles.append(Particle(i, 1 / num_particles, deepcopy(initial_pose), map_params=map_params))
        self.weights = np.array([1/num_particles for i in range(num_particles)])
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

        self.res = self.particles[0].map.res
        self.size_x = self.particles[0].map.size_x
        self.size_y = self.particles[0].map.size_y
        self.center_x = self.particles[0].map.center_x
        self.center_y = self.particles[0].map.center_y
        self.max_range = 10
        self.log_free = np.log(0.4 / (1 - 0.4))
        self.log_occupied = np.log(0.8 / (1 - 0.8))

        self.weights_log = []
        self.score_log = []
        self.max_diff = 0.3

    def update_particles(self, range_sensor, odometry):
        start = time.time()
        diff_log = []
        normalizers = []
        observation_likelihoods = []
        motion_likelihoods = []
        prev_pose = [deepcopy(p.pose) for p in self.particles]
        prev_weight = [w for w in self.weights]
        prev_maps = [p.map.log_prob_map for p in self.particles]
        score_log = []
        for p in self.particles:
            odometry_pose = self.odometry_update(p.get_pose(),odometry)
            scan_pose, score = self.scan_matcher.scan_match(self.rays, range_sensor, odometry_pose, p.map)
            if np.linalg.norm(scan_pose[0:2] - odometry_pose[0:2]) > self.max_diff:
                scan_pose[0:2] = odometry_pose[0:2]
            score_log.append(score)
            diff_log.append(scan_pose - odometry_pose)
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

            observation_likelihoods.append(n_obs)
            motion_likelihoods.append(n_mot)
            mean = mean / normalizer
            covariance = np.zeros([3, 3])

            for (i, s) in enumerate(samples):
                diff = diff_covariance(s, mean)
                covariance += diff * diff.transpose() * likelihoods[i]

            covariance = covariance / normalizer

            normalizers.append(normalizer.item())

            try:
                pose = np.random.multivariate_normal(mean.squeeze(), covariance).reshape(3, 1)
                p.weight = p.weight * normalizer
            except:
                pose = scan_pose
                self.weight = 1e-200

            if self.counter < 3:
                pose = p.get_pose()

            pose[2] = pose[2] % (2 * np.pi)
            p.pose = deepcopy(pose)
            p.map.integrate_scan(p.pose, range_sensor, self.rays)
            p.trajectory.append(p.pose.copy())
            p.update_loop_graph()


        """
        process = []
        q = mp.Queue()
        for i, p in enumerate(self.particles):
            pro = mp.Process(target=integrate_scan_mp, args=(i,q,p.map.log_prob_map, p.get_pose(), range_sensor.copy(), self.rays.copy(), self.center_x , self.center_y,
                                                             self.log_free, self.log_occupied, self.max_range, self.res, self.size_x, self.size_y))
            pro.start()
            process.append(pro)
        a = [q.get() for _ in process]
        [x.join() for x in process]
        for m in a:
            self.particles[m[0]].map.log_prob_map = m[1]
        """
        end = time.time()


        print("Time Update: {:.6f}".format(end-start))
        self.normalize_weights()
        self.weights_log.append(self.weights.copy())
        self.score_log.append(score_log)
        #if self.counter % 15 == 0:
        #    self.visualize_all_particles()
        if self.n_eff < self.threshold_resampling:
            self.resampling_counter += 1
            print("Resampling particles! Resampling num:", self.resampling_counter )
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

    def visualize_all_particles(self):
        total_map = np.zeros(self.particles[self.best_particle].map.log_prob_map.shape)
        for p in self.particles:
            total_map += p.map.log_prob_map
        prob_map = np.exp(total_map)/(1+np.exp(total_map))
        plt.figure()
        plt.imshow(prob_map.T,"Greys", origin="lower", extent=[-self.size_x/2*self.res,self.size_x/2*self.res,-self.size_y/2*self.res,self.size_y/2*self.res])
        for p in self.particles:
            plt.plot(p.pose[0], p.pose[1],"x", markersize=4)
        plt.pause(10)
        plt.close(plt.gcf())

    def visualize(self):
        self.particles[self.best_particle].visualize()

    def visualize_loop_graph(self):
        self.particles[self.best_particle].visualize_loop_graph()

    def get_time_entry(self):
        if self.counter % 10 == 0:
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

    def visualize_weights(self):
        weights = {}
        score = {}
        for i in range(self.num_particles):
            weights[i] = [w[i] for w in self.weights_log]
            score[i] = [s[i] for s in self.score_log]

        t = np.arange(0,len(self.weights_log))
        plt.figure()
        for i in range(self.num_particles):
            plt.plot(t, weights[i])
        plt.grid()
        plt.legend([str(i) for i in range(self.num_particles)])
        plt.figure()
        for i in range(self.num_particles):
            plt.plot(t, score[i])
        plt.grid()
        plt.legend([str(i) for i in range(self.num_particles)])

def diff_covariance(sample,mean):
    diff = np.zeros([3,1])
    diff[0:2] = sample[0:2]-mean[0:2]
    diff[2] = ssa(sample[2],mean[2])
    return diff


def integrate_scan_mp(id, q,map, pose, measurments, ray_vectors, center_x, center_y, log_free, log_occupied, max_range, res, size_x, size_y):
    ray_vectors = rot_matrix_2d(pose[2]) @ ray_vectors[0:2, :]
    for i in range(measurments.size):
        meas = measurments[i]
        ray = ray_vectors[0:2, i].reshape(2,1)
        if meas == np.inf:
            end_point = pose[0:2] + ray * max_range
        else:
            end_point = pose[0:2] + ray * meas

        end_point_cell = [int(floor(end_point[0]/res)) + center_x, int(floor(end_point[1]/res))+center_y]
        if meas != np.inf:
            if not (end_point_cell[0] < 0 or end_point_cell[0] > size_x or end_point_cell[1] < 0 or end_point_cell[1] >= size_y):
                map[end_point_cell[0],end_point_cell[1]] += log_occupied
        free_cells = grid_traversal(pose[0:2], end_point, res, center_x, center_y, size_x, size_y)
        for cell in free_cells:
            if not (cell[0] < 0 or cell[0] >= size_x or cell[1] < 0 or cell[1] >= size_y):
                map[cell[0],cell[1]] += log_free
    q.put((id, map))



def grid_traversal(start_point, end_point, res, center_x, center_y, size_x, size_y):
    visited_cells = []
    current_cell = [int(floor(start_point[0] / res)) + center_x, int(floor(start_point[1] / res)) + center_y]
    last_cell = [int(floor(end_point[0]/res)) + center_x, int(floor(end_point[1]/res))+center_y]
    ray = (end_point-start_point)
    ray = ray/np.linalg.norm(ray)

    stepX = 1 if ray[0] >= 0 else -1
    stepY = 1 if ray[1] >= 0 else -1

    next_cell_boundary_x = (current_cell[0]-center_x + stepX)*res
    next_cell_boundary_y = (current_cell[1]-center_y + stepY)*res

    tmaxX = (next_cell_boundary_x-start_point[0])/ray[0] if ray[0] != 0 else np.inf
    tmaxY = (next_cell_boundary_y-start_point[1])/ray[1] if ray[1] != 0 else np.inf

    tdeltaX = res/ray[0]*stepX if ray[0] != 0 else np.inf
    tdeltaY = res/ray[1]*stepY if ray[1] != 0 else np.inf

    neg_ray = False
    diff = np.zeros(2,np.int)
    if current_cell[0] != last_cell[0] and ray[0] < 0:
        diff[0] -= 1
        neg_ray = True
    if current_cell[1] != last_cell[1] and ray[1] < 0:
        diff[1] -= 1
        neg_ray = True
    if neg_ray:
        visited_cells.append(current_cell.copy())
        current_cell += diff

    while current_cell[0] != last_cell[0] or current_cell[1] != last_cell[1]:
        visited_cells.append(current_cell.copy())
        if tmaxX <= tmaxY:
            current_cell[0] += stepX
            tmaxX += tdeltaX
        else:
            current_cell[1] += stepY
            tmaxY += tdeltaY
    return visited_cells