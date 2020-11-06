import numpy as np
import matplotlib.pyplot as plt
from sensor.lidar_sensor import LidarSensor
from queue import Queue

class SharedMap():
    def __init__(self,base_id, maps, initial_poses):
        self.min_frontier_length = 10
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
        self.free_threshold = np.log(0.2/(1-0.2))
        self.rays = LidarSensor(np.array([0, 0, 0.1]), np.array([0, 0, 0]), num_rays=180).ray_vectors
        self.max_range = 10

    def merge_map(self,maps):
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

        starting_cell = [self.center_x, self.center_y]

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
        return frontiers

    def compute_frontier_points(self):
        frontiers = self.compute_frontiers()
        frontiers_f = []
        for f in frontiers:
            if len(f) >= self.min_frontier_length:
                frontiers_f.append(np.array(f).T)
        mean = []
        for f in frontiers_f:
            mean.append(np.mean(f,axis=1))
        return mean

    def get_waypoint_exploration(self, idle_drones, active_drones):
        points = self.compute_frontier_points()
        information_gain = []
        self.get_occupancy_grid()
        for i in range(len(points)):
            information_gain.append({"point": points[i], "observable_cells": self.calculate_observable_cells(points[i])})
        assigned_cells = np.zeros(self.log_prob_map.shape)
        for a in active_drones.keys():
            observed = self.calculate_observable_cells(active_drones[a])
            for c in observed:
                assigned_cells[c[0],c[1]] = 1

        def sort_func(e):
            count = 0
            for c in e["observable_cells"]:
                if assigned_cells[c[0], c[1]] == 0:
                    count += 1
            return count

        information_gain.sort(reverse=True,key=sort_func)
        return information_gain

    def calculate_observable_cells(self, pose):
        observed = np.zeros(self.log_prob_map.shape)
        unknowns = []
        pose = pose[0:2].reshape(2,1)
        for i in range(self.rays.shape[1]):
            ray = self.rays[0:2,i].reshape(2,1)
            end = pose[0:2] + self.max_range/self.res*ray
            unknown_cells = self.grid_traversal(pose[0:2],end)
            for c in unknown_cells:
                if observed[c[0],c[1]] == 0:
                    observed[c[0],c[1]] = 1
                    unknowns.append(c)
        return unknowns
    
    def distance
    
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

    def get_cell(self, point):
        x = np.int(np.floor(point[0]))
        y = np.int(np.floor(point[1]))
        return np.array([x,y]).reshape(2,1)

    def compute_coordinate_local_map(self, id, coordinate):
        x = (np.floor(coordinate[0]) - self.center_x)*self.res - self.delta_poses[id][0]
        y = (np.floor(coordinate[1]) - self.center_y)*self.res - self.delta_poses[id][1]
        return np.array([x,y,0],np.float).reshape(3,1)

    def get_occupancy_grid(self):
        occ_grid = np.zeros(self.log_prob_map.shape)
        occ_grid[self.log_prob_map > self.occ_threshold] = -1
        occ_grid[self.log_prob_map < self.free_threshold] = 1
        self.occ_grid = occ_grid
        return occ_grid.copy()

    def compute_cell(self, id, coord):
        x = np.floor(coord[0]/self.res) + self.delta_x[id] + self.center_x
        y = np.floor(coord[1]/self.res) + self.delta_y[id] + self.center_y
        return np.array([x,y])

    def visualize_frontiers(self, frontiers):
        f_cells = np.zeros(self.log_prob_map.shape)
        x = []
        y = []
        for f in frontiers:
            if len(f) < 3:
                continue
            for c in f:
                x.append(c[0])
                y.append(c[1])
        f_cells[x,y] = 1
        plt.figure("Frontiers")
        plt.imshow(f_cells.T,"Greys", origin="lower",)
        plt.pause(0.1)

    def convert_grid_to_prob(self):
        exp_grid = np.exp(self.log_prob_map)
        return exp_grid/(1+exp_grid)

    def get_free_cells(self):
        cells = np.where(self.log_prob_map < np.log(0.2 / (1 - 0.2)))
        cells = np.concatenate([cells[0].reshape(-1,1),cells[1].reshape(-1,1)],axis=1)
        return cells

    def init_plot(self, axis):
        im = axis.imshow(self.convert_grid_to_prob().transpose(), "Greys", origin="lower",
                         extent=[-self.size_x_merged / 2 * self.res, self.size_x_merged / 2 * self.res, -self.size_y_merged / 2 * self.res,
                                 self.size_y_merged / 2 * self.res])
        return im

    def update_plot(self, im):
        im.set_data(self.convert_grid_to_prob().transpose())
        im.autoscale()
        return im

    def visualize(self):
        plt.figure()
        plt.imshow(self.convert_grid_to_prob().transpose(), "Greys", origin="lower",
                         extent=[-self.size_x_merged / 2 * self.res, self.size_x_merged / 2 * self.res, -self.size_y_merged / 2 * self.res,
                                 self.size_y_merged / 2 * self.res])
        plt.show()