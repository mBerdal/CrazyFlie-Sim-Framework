import numpy as np
from slam.gridslam import GridSLAM
from sensor.lidar_sensor import LidarSensor
from sensor.odometry_sensor import OdometrySensor
from logger.loggable import Loggable
from utils.misc import compute_dist_loop_graph
import slam.params as params
from planning.path_planning import AStar

class Slammer(Loggable):
    """
    Wrapper class for the SLAM algorithm for easier implementation with the controllers.
    """
    def __init__(self, id, num_particles, rays, **kwargs):
        self.id = id
        self.map_params = kwargs.get("map_params",params.map_params)
        self.obs_params = kwargs.get("obs_params",params.obs_params)
        self.slam_params = kwargs.get("slam_params",params.slam_params)
        self.odometry_params = kwargs.get("odometry_params", params.odometry_params)
        self.particle_params = kwargs.get("particle_params", params.particle_params)
        self.scan_params = kwargs.get("scan_params", params.scan_match_params)

        self.num_particles = num_particles
        self.initial_pose = kwargs.get("initial_pose",np.zeros([3,1]))
        self.rays = rays

        self.slam = GridSLAM(id, self.num_particles, self.initial_pose, self.rays, particle_params=self.particle_params,
                             map_params=self.map_params, scan_match_params=self.scan_params, obs_params=self.obs_params,
                             odometry_params=self.odometry_params, **self.slam_params)

        self.prev_odometry = np.zeros([3,1])

        self.update_time = kwargs.get("update_time", 0.50)
        self.update_translation = kwargs.get("update_translation", 0.50)
        self.update_rotation = kwargs.get("update_rotation", np.deg2rad(20))

        self.time_counter = 0

        self.occupied_threshold = 0.8
        self.log_occupied_threshold = np.log(self.occupied_threshold/(1-self.occupied_threshold))

    def update(self, sensor_data, time_step):
        lidar_data = None
        odometry_data = None
        for sd in sensor_data:
            if sd["type"] == LidarSensor:
                lidar_data = sd["reading"]
            if sd["type"] == OdometrySensor:
                odometry_data = sd["reading"]

        update_flag = False
        if lidar_data is not None and odometry_data is not None:
            diff_translation = np.linalg.norm(odometry_data[0:2]-self.prev_odometry[0:2])
            diff_rotation = np.abs(odometry_data[2]-self.prev_odometry[2])
            self.increment_time_counter(time_step)
            if self.time_counter >= self.update_time:
                update_flag = True
            if diff_translation >= self.update_translation:
                update_flag = True
            if diff_rotation >= self.update_rotation:
                update_flag = True

            if update_flag:
                print("Updating SLAM, Id:", self.id)
                odometry = np.concatenate([odometry_data.reshape(1, 3), self.prev_odometry.reshape(1,3)], axis=0)
                self.slam.update_particles(lidar_data,odometry)

                self.reset_time_counter()
                self.increment_time_counter(time_step)
                self.prev_odometry = odometry_data.copy()
        return update_flag

    def increment_time_counter(self, time_step):
        self.time_counter += time_step

    def reset_time_counter(self):
        self.time_counter = 0

    def get_map(self):
        return self.slam.get_best_map()

    def get_pose(self):
        return self.slam.get_best_pose()

    def get_trajectory(self):
        return self.slam.get_trajectory()

    def check_loop_closure(self, min_t_dist, max_d_dist):
        graph, current_node = self.slam.get_loop_graph()
        if len(graph) < min_t_dist:
            return []
        t_dist = compute_dist_loop_graph(graph, current_node)
        pose = self.get_pose()
        d_dist = [np.linalg.norm(g.pos - pose[0:2]).item() for g in graph]

        indicator = [d_dist[i] <= max_d_dist and t_dist[i] >= min_t_dist for i in range(len(graph))]
        map_res = self.get_map().res
        occ_grid = self.get_map().get_occ_grid(0.7,0.3)
        distances = []
        for i in range(len(graph)):
            if indicator[i]:
                pose_cell = self.get_map().world_coordinate_to_grid_cell(self.get_pose())
                g_cell = self.get_map().world_coordinate_to_grid_cell(graph[i].pos)
                a = AStar(pose_cell, g_cell, occ_grid)
                if a.init_sucsess:
                    res = a.planning()
                    if res is not None:
                        distances.append(res[1]*map_res)
                    else:
                        distances.append(np.inf)
                else:
                    distances.append(np.inf)
            else:
                distances.append(np.inf)
        loops = []
        for i, d in enumerate(distances):
            if d <= max_d_dist:
                loops.append((graph[i].pos, d, t_dist[i]))
        return loops


    def get_dist_grid(self, radius):
        pose = self.get_pose()
        map = self.get_map()
        lower_x = np.int(
            np.floor((pose[0] - radius) / map.res)) + map.center_x
        lower_y = np.int(
            np.floor((pose[1] - radius) / map.res)) + map.center_y

        upper_x = np.int(
            np.floor((pose[0] + radius) / map.res)) + map.center_x + 1
        upper_y = np.int(
            np.floor((pose[1] + radius) / map.res)) + map.center_y + 1

        lower_x = max([lower_x, 0])
        lower_y = max([lower_y, 0])

        upper_x = min([upper_x, map.size_x])
        upper_y = min([upper_y, map.size_y])

        lower_x_coord = (lower_x - map.center_x) * map.res
        lower_y_coord = (lower_y - map.center_y) * map.res

        prob_field = map.log_prob_map[lower_x:upper_x, lower_y:upper_y].copy()

        binary_field = prob_field > self.log_occupied_threshold
        occupied_cells_x, occupied_cells_y = np.where(binary_field)

        dist_field = np.inf * np.ones(prob_field.shape)
        dist_field[occupied_cells_x, occupied_cells_y] = 0

        x = np.arange(0, dist_field.shape[1]) * map.res
        y = np.arange(0, dist_field.shape[0]) * map.res

        xx, yy = np.meshgrid(x, y)
        for i in range(occupied_cells_x.shape[0]):
            cell_x = occupied_cells_x[i]
            cell_y = occupied_cells_y[i]
            dist_occupied = np.sqrt((xx - xx[cell_x, cell_y]) ** 2 + (yy - yy[cell_x, cell_y]) ** 2)
            dist_field = np.minimum(dist_field, dist_occupied)

        return {"lower_x": lower_x_coord, "lower_y": lower_y_coord, "res": map.res, "grid": dist_field}

    def get_info_entry(self):
        return self.slam.get_info_entry()

    def get_time_entry(self):
        return self.slam.get_time_entry()

    def generate_time_entry(self):
        return self.slam.generate_time_entry()

    def visualize(self):
        self.slam.visualize()

    def init_plot(self,axes):
        obj = self.slam.init_plot(axes)
        return obj

    def update_plot(self, objects):
        return self.slam.update_plot(objects)