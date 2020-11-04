import numpy as np
import matplotlib.pyplot as plt
from queue import Queue

class SharedMap():
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

        self.merged_map = np.zeros([self.size_x_merged,self.size_y_merged])

        self.merged_map_center_y = np.int(np.floor(self.size_x_merged/2))
        self.merged_map_center_x = np.int(np.floor(self.size_y_merged/2))

        self.center_x = self.base_center_x + self.delta_x[base_id]
        self.center_y = self.base_center_y + self.delta_y[base_id]

        self.occ_threshold = np.log(0.7/(1-0.7))

    def merge_map(self,maps):
        self.merged_map = np.zeros([self.size_x_merged,self.size_y_merged])
        for i in maps.keys():
            if maps[i] is None:
                continue
            self.merged_map[self.start_x[i]:self.end_x[i],self.start_y[i]:self.end_y[i]] += maps[i].log_prob_map
        return self.merged_map

    def compute_frontiers(self):
        cells = np.zeros(self.merged_map.shape)
        cells[self.merged_map > self.occ_threshold] = -1
        cells[self.merged_map == 0] = 1

        starting_cell = [self.center_x, self.center_y]

        q = Queue(0)
        q.put(starting_cell)

        visited = np.zeros(self.merged_map.shape)

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

    def get_waypoint_exploration(self):
        wps = {}
        points = self.compute_frontier_points()
        points = np.random.permutation(points)
        for ind, i in enumerate(self.ids):
            wps[i] = self.compute_coordinate_local_map(i,points[ind])
        return wps

    def compute_coordinate_local_map(self, id, coordinate):
        x = (np.floor(coordinate[0]) - self.center_x)*self.res - self.delta_poses[id][0]
        y = (np.floor(coordinate[1]) - self.center_y)*self.res - self.delta_poses[id][1]
        return np.array([x,y,0],np.float).reshape(3,1)

    def visualize_frontiers(self, frontiers):
        f_cells = np.zeros(self.merged_map.shape)
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
        exp_grid = np.exp(self.merged_map)
        return exp_grid/(1+exp_grid)

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