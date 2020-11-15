import numpy as np
import matplotlib.pyplot as plt
from sensor.lidar_sensor import LidarSensor
from queue import Queue

from logger.loggable import Loggable
from logger.log_entry import LogEntry

from inspect import getmodule
from copy import deepcopy


class SharedMap(Loggable):
    def __init__(self,base_id, maps, initial_poses):
        self.min_frontier_length = 20
        self.ids = maps.keys()
        self.base_id = base_id
        self.res = maps[base_id].res

        self.initial_poses = initial_poses

        self.delta_x = {}
        self.delta_y = {}

        self.base_pose = initial_poses[base_id]
        self.base_center_x = maps[base_id].center_x
        self.base_center_y = maps[base_id].center_y

        self.delta_poses = {}
        for i in self.ids:
            self.delta_poses[i] = initial_poses[i]-self.base_pose

        self.size_x = {}
        self.size_y = {}

        for i in self.ids:
            self.delta_x[i] = np.int(np.floor((initial_poses[i][0]-self.base_pose[0])/self.res))
            self.delta_y[i] = np.int(np.floor((initial_poses[i][1]-self.base_pose[1])/self.res))

            self.size_x[i] = maps[i].size_x
            self.size_y[i] = maps[i].size_y

        self.max_delta_x = max(self.delta_x.values())
        self.min_delta_x = min(self.delta_x.values())
        self.max_delta_y = max(self.delta_y.values())
        self.min_delta_y = min(self.delta_y.values())

        self.delta_x = {k: v - self.min_delta_x for k,v in self.delta_x.items()}
        self.delta_y = {k: v - self.min_delta_y for k,v in self.delta_y.items()}

        self.start_x = {}
        self.start_y = {}
        self.end_x = {}
        self.end_y = {}

        for i in self.ids:
            self.start_x[i] = self.delta_x[i]
            self.start_y[i] = self.delta_y[i]
            self.end_x[i] = self.delta_x[i]+self.size_x[i]
            self.end_y[i] = self.delta_y[i]+self.size_y[i]

        self.size_x_merged = max(self.end_x.values())
        self.size_y_merged = max(self.end_y.values())

        self.log_prob_map = np.zeros([self.size_x_merged, self.size_y_merged])
        self.occ_grid = np.zeros([self.size_x_merged,self.size_y_merged])

        self.merged_map_center_y = np.int(np.floor(self.size_x_merged/2))
        self.merged_map_center_x = np.int(np.floor(self.size_y_merged/2))

        self.center_x = self.base_center_x + self.delta_x[base_id]
        self.center_y = self.base_center_y + self.delta_y[base_id]

        self.occ_threshold = np.log(0.7/(1-0.7))
        self.free_threshold = np.log(0.3/(1-0.3))
        self.rays = LidarSensor(np.array([0, 0, 0.1]), np.array([0, 0, 0]), num_rays=180).ray_vectors
        self.max_range = 10

        self.frontiers = None
        self.unreachable_frontiers = []
        self.dist_unreachable = 10

        self.counter = 0

    def merge_map(self, maps):
        self.counter += 1
        self.log_prob_map = np.zeros([self.size_x_merged, self.size_y_merged])
        for i in maps.keys():
            if maps[i] is None:
                continue
            self.log_prob_map[self.start_x[i]:self.end_x[i], self.start_y[i]:self.end_y[i]] += maps[i].log_prob_map
        return self.log_prob_map

    def compute_frontiers(self):
        cells = np.zeros(self.log_prob_map.shape)
        cells[self.log_prob_map > self.occ_threshold] = -1
        cells[self.log_prob_map == 0] = 1

        starting_cell = self.cell_from_coordinate_local_map(self.base_id,self.base_pose)

        q = Queue(0)
        q.put(starting_cell)

        visited = np.zeros(self.log_prob_map.shape)

        frontiers = list()

        neighbors_4 = [[-1,0], [0,1], [0,-1], [1,0]]
        neighbors_8 = [[-1,1],[0,1],[1,1],[-1,0],[1,-0],[-1,-1],[0,-1],[1,-1]]

        def check_frontier_cell(cell):
            for ne in neighbors_4:
                try:
                    if cells[cell[0]+ne[0],cell[1]+ne[1]] == 1:
                        return True
                except IndexError:
                    pass
            return False

        while not q.empty():
            c = q.get()
            if visited[c[0],c[1]]:
                continue

            if check_frontier_cell(c):
                new_frontier = []
                qf = Queue(0)
                qf.put(c)

                while not qf.empty():
                    cf = qf.get()
                    if visited[cf[0],cf[1]]:
                        continue
                    if check_frontier_cell(cf):
                        for n in neighbors_8:
                            x = cf[0] + n[0]
                            y = cf[1] + n[1]
                            try:
                                if cells[x,y] == 0 and not visited[x,y]:
                                    qf.put([x,y])
                            except IndexError:
                                pass
                        new_frontier.append(cf)
                    visited[cf[0],cf[1]] = 1
                frontiers.append(new_frontier)
            else:
                for n in neighbors_4:
                    x = c[0] + n[0]
                    y = c[1] + n[1]
                    try:
                        if cells[x,y] == 0 and not visited[x,y]:
                            q.put([x,y])
                    except IndexError:
                        pass
                visited[c[0],c[1]] = 1
        self.frontiers = frontiers

    def compute_occupancy_grid(self):
        occ_grid = np.zeros(self.log_prob_map.shape)
        occ_grid[self.log_prob_map > self.occ_threshold] = -1
        occ_grid[self.log_prob_map < self.free_threshold] = 1
        self.occ_grid = occ_grid

    def grid_traversal(self, start_point, end_point):
        visited_cells = []
        current_cell = self.get_cell(start_point)
        last_cell = self.get_cell(end_point)
        ray = (end_point - start_point)
        ray = ray / np.linalg.norm(ray)

        stepX = 1 if ray[0] >= 0 else -1
        stepY = 1 if ray[1] >= 0 else -1

        next_cell_boundary_x = (current_cell[0] + stepX)
        next_cell_boundary_y = (current_cell[1] + stepY)

        tmaxX = (next_cell_boundary_x - start_point[0]) / ray[0] if ray[0] != 0 else np.inf
        tmaxY = (next_cell_boundary_y - start_point[1]) / ray[1] if ray[1] != 0 else np.inf

        tdeltaX = 1 / ray[0] * stepX if ray[0] != 0 else np.inf
        tdeltaY = 1 / ray[1] * stepY if ray[1] != 0 else np.inf

        neg_ray = False
        diff = np.zeros([2,1], np.int)

        if current_cell[0] != last_cell[0] and ray[0] < 0:
            diff[0] -= 1
            neg_ray = True
        if current_cell[1] != last_cell[1] and ray[1] < 0:
            diff[1] -= 1
            neg_ray = True
        if neg_ray:
            visited_cells.append(current_cell.copy())
            current_cell += diff

        while np.any(current_cell != last_cell):
            if self.occ_grid[current_cell[0],current_cell[1]] == 0:
                visited_cells.append(current_cell.copy())
            if tmaxX <= tmaxY:
                current_cell[0] += stepX
                tmaxX += tdeltaX
            else:
                current_cell[1] += stepY
                tmaxY += tdeltaY
            try:
                if self.occ_grid[current_cell[0],current_cell[1]] == -1:
                    break
            except IndexError:
                return []
        return visited_cells

    def check_collision(self, start, end):
        ray = (end - start)
        ray = ray / np.linalg.norm(ray)

        step_x = 1 if ray[0] >= 0 else -1
        step_y = 1 if ray[1] >= 0 else -1

        current_cell = start.copy()

        next_cell_boundary_x = (current_cell[0] + step_x)
        next_cell_boundary_y = (current_cell[1] + step_y)

        tmax_x = (next_cell_boundary_x - start[0]) / ray[0] if ray[0] != 0 else np.inf
        tmax_y = (next_cell_boundary_y - start[1]) / ray[1] if ray[1] != 0 else np.inf

        tdelta_x = 1 / ray[0] * step_x if ray[0] != 0 else np.inf
        tdelta_y = 1 / ray[1] * step_y if ray[1] != 0 else np.inf

        while np.any(current_cell != end):
            if tmax_x <= tmax_y:
                current_cell[0] += step_x
                tmax_x += tdelta_x
            else:
                current_cell[1] += step_y
                tmax_y += tdelta_y
            if self.occ_grid[current_cell[0], current_cell[1]] == -1:
                return True
        return False

    def coordinate_from_cell_local_map(self, id, coordinate):
        x = (np.floor(coordinate[0]) - self.center_x)*self.res - self.delta_poses[id][0]
        y = (np.floor(coordinate[1]) - self.center_y)*self.res - self.delta_poses[id][1]
        return np.array([x,y,0],np.float).reshape(3,1)

    def cell_from_coordinate_local_map(self, id, coordinate):
        x = np.int(np.floor(coordinate[0] / self.res)) + self.center_x + self.delta_x[id]
        y = np.int(np.floor(coordinate[1] / self.res)) + self.center_y + self.delta_y[id]
        return np.array([x,y]).reshape(2,1)

    def convert_log_to_prob(self):
        exp_grid = np.exp(self.log_prob_map)
        return exp_grid/(1+exp_grid)

    def get_frontier_points(self, min_frontier_length):
        frontiers_f = []
        for f in self.frontiers:
            if len(f) >= min_frontier_length:
                frontiers_f.append(np.array(f,dtype=np.float).squeeze().T)
        mean = []
        for f in frontiers_f:
            mean_f = np.mean(f,axis=1)
            x = np.int(np.floor(mean_f[0]))
            y = np.int(np.floor(mean_f[1]))
            mean.append(np.array([x,y],dtype=np.int))
        return mean

    def get_observable_cells_from_pos(self, cell_pos, max_range):
        observed = np.zeros(self.log_prob_map.shape)
        unknowns = []
        cell_pos = cell_pos[0:2].reshape(2, 1)
        for i in range(self.rays.shape[1]):
            ray = self.rays[0:2,i].reshape(2,1)
            end = cell_pos[0:2] + max_range / self.res * ray
            unknown_cells = self.grid_traversal(cell_pos[0:2], end)
            for c in unknown_cells:
                if observed[c[0],c[1]] == 0:
                    observed[c[0],c[1]] = 1
                    unknowns.append(c)
        return unknowns

    def get_occupancy_grid(self, pad=0):
        occ_grid = np.zeros(self.log_prob_map.shape)
        occ_grid[self.log_prob_map > self.occ_threshold] = -1
        occ_grid[self.log_prob_map < self.free_threshold] = 1
        if pad > 0:
            neighbors = [[x, y] for x in range(-pad, pad + 1) for y in range(-pad, pad + 1)]
            occupied_cells_x, occupied_cells_y = np.where(occ_grid == -1)
            for i in range(len(occupied_cells_x)):
                for n in neighbors:
                    x = occupied_cells_x[i] + n[0]
                    y = occupied_cells_y[i] + n[1]
                    occ_grid[x, y] = -1
        return occ_grid

    def get_free_cells(self):
        cells = np.where(self.occ_grid == 1)
        cells = np.concatenate([cells[0].reshape(-1,1),cells[1].reshape(-1,1)],axis=1)
        return cells

    def get_map(self):
        return deepcopy(self.log_prob_map)

    def get_size(self):
        return [self.size_x_merged,self.size_y_merged]

    def get_cell(self, point):
        x = np.int(np.floor(point[0]))
        y = np.int(np.floor(point[1]))
        return np.array([x,y]).reshape(2,1)

    def init_plot(self, axis):
        im = axis.imshow(self.convert_log_to_prob().transpose(), "Greys", origin="lower",
                         extent=[-self.size_x_merged / 2 * self.res, self.size_x_merged / 2 * self.res, -self.size_y_merged / 2 * self.res,
                                 self.size_y_merged / 2 * self.res])
        return im

    def update_plot(self, im):
        im.set_data(self.convert_log_to_prob().transpose())
        im.autoscale()
        return im

    def visualize(self):
        plt.figure()
        plt.imshow(self.convert_log_to_prob().transpose(), "Greys", origin="lower",
                   extent=[-self.size_x_merged / 2 * self.res, self.size_x_merged / 2 * self.res, -self.size_y_merged / 2 * self.res,
                                 self.size_y_merged / 2 * self.res])
        plt.show()

    def get_time_entry(self):
        if self.counter % 10 == 0:
            return LogEntry(
                counter=self.counter,
                map=self.get_map()
            )
        else:
            return LogEntry(
                counter=self.counter,
            )

    def generate_time_entry(self):
        if self.counter % 10 == 0:
            return LogEntry(
                counter=self.counter,
                map=self.get_map()
            )
        else:
            return LogEntry(
                counter=self.counter,
            )

    def get_info_entry(self):
        return LogEntry(
            module=getmodule(self).__name__,
            cls=type(self).__name__,
            map_res=self.res,
            map_size_x=self.size_x_merged,
            map_size_y=self.size_y_merged,
        )