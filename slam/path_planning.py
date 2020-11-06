import numpy as np
import matplotlib.pyplot as plt

class RRT:
    class Node:
        def __init__(self, coord):
            self.coord = coord
            self.parent = None
            self.path_x = []
            self.path_y = []

    def __init__(self, start, target, map, free_cells):
        self.nodes = []
        self.start = self.Node(np.array(start,np.int).reshape(2,1))
        self.target = self.Node(np.array(target,np.int).reshape(2,1))

        self.expand_dist = 15
        self.target_sample_rate = 0.20
        self.max_iter = 500
        self.map = map
        self.nodes_list = [self.start]

        self.free_cells = free_cells
        self.num_free_cells = len(self.free_cells)

        self.obstacle_t = 0.05

    def planning(self):
        iter = 0

        while iter < self.max_iter:
            rnd = self.get_random_node()
            ind_nearest = self.find_nearest_node(rnd)
            neigh = self.nodes_list[ind_nearest]

            node = self.steer(neigh, rnd)
            if node is not None:
                node.parent = neigh
                self.nodes_list.append(node)
            else:
                continue
            dist_goal = self.dist_to_goal(node)

            if dist_goal < self.expand_dist:
                if not self.check_collision(node, self.target):
                    self.target.parent = node
                    return self.get_path_to_target()
            iter += 1
        return None

    def dist_to_goal(self, node):
        dist = np.linalg.norm(node.coord-self.target.coord)
        return dist

    def find_nearest_node(self, node):
        dist = [np.linalg.norm(node.coord-n.coord) for n in self.nodes_list]
        return np.argmin(dist).item()

    def get_random_node(self):
        if np.random.rand() > self.target_sample_rate:
            i = np.random.random_integers(0,self.num_free_cells-1)
            return self.Node(self.free_cells[i].reshape(2, 1).copy())
        else:
            return self.Node(self.target.coord.copy())

    def steer(self, start, end):
        ray = (end.coord - start.coord)
        ray = ray / np.linalg.norm(ray)

        step_x = 1 if ray[0] >= 0 else -1
        step_y = 1 if ray[1] >= 0 else -1

        current_cell = start.coord.copy()

        next_cell_boundary_x = (current_cell[0] + step_x)
        next_cell_boundary_y = (current_cell[1] + step_y)

        tmax_x = (next_cell_boundary_x - start.coord[0]) / ray[0] if ray[0] != 0 else np.inf
        tmax_y = (next_cell_boundary_y - start.coord[1]) / ray[1] if ray[1] != 0 else np.inf

        tdelta_x = 1 / ray[0] * step_x if ray[0] != 0 else np.inf
        tdelta_y = 1 / ray[1] * step_y if ray[1] != 0 else np.inf

        neg_ray = False
        diff = np.zeros([2,1], np.int)
        """
        if current_cell[0] != end.coord[0] and ray[0] < 0:
            diff[0] -= 1
            neg_ray = True
        if current_cell[1] != end.coord[1] and ray[1] < 0:
            diff[1] -= 1
            neg_ray = True
        if neg_ray:
            current_cell += diff
        """
        i = 0
        while np.any(current_cell != end.coord) and i < self.expand_dist:
            if tmax_x <= tmax_y:
                current_cell[0] += step_x
                tmax_x += tdelta_x
            else:
                current_cell[1] += step_y
                tmax_y += tdelta_y
            if self.map[current_cell[0],current_cell[1]] == -1:
                return None
            i += 1

        return self.Node(current_cell)

    def check_collision(self, start, end):
        ray = (end.coord - start.coord)
        ray = ray / np.linalg.norm(ray)

        step_x = 1 if ray[0] >= 0 else -1
        step_y = 1 if ray[1] >= 0 else -1

        current_cell = start.coord.copy()

        next_cell_boundary_x = (current_cell[0] + step_x)
        next_cell_boundary_y = (current_cell[1] + step_y)

        tmax_x = (next_cell_boundary_x - start.coord[0]) / ray[0] if ray[0] != 0 else np.inf
        tmax_y = (next_cell_boundary_y - start.coord[1]) / ray[1] if ray[1] != 0 else np.inf

        tdelta_x = 1 / ray[0] * step_x if ray[0] != 0 else np.inf
        tdelta_y = 1 / ray[1] * step_y if ray[1] != 0 else np.inf

        neg_ray = False
        diff = np.zeros([2, 1], np.int)
        """
        if current_cell[0] != end.coord[0] and ray[0] < 0:
            diff[0] -= 1
            neg_ray = True
        if current_cell[1] != end.coord[1] and ray[1] < 0:
            diff[1] -= 1
            neg_ray = True
        if neg_ray:
            current_cell += diff
        """
        i = 0
        while np.any(current_cell != end.coord):
            i += 1
            if tmax_x <= tmax_y:
                current_cell[0] += step_x
                tmax_x += tdelta_x
            else:
                current_cell[1] += step_y
                tmax_y += tdelta_y
            if self.map[current_cell[0], current_cell[1]] == -1:
                return True
        return False


    def get_path_to_target(self):
        wps = [self.target]
        current_node = self.target.parent
        prev_node = self.target
        i = 0
        while current_node is not self.start:
            if not self.check_collision(wps[i], current_node):
                prev_node = current_node
                current_node = current_node.parent
            else:
                wps.append(prev_node)
                i += 1
            if i > 100:
                return None
        wps = [w.coord for w in wps]
        wps.reverse()
        return wps

    def visualize_graph(self):
        plt.figure()
        plt.imshow(self.map.T,"Greys", origin="lower")

        for node in self.nodes_list:
            if node.parent:
                x = [node.coord[0], node.parent.coord[0]]
                y = [node.coord[1], node.parent.coord[1]]
                plt.plot(x,y,"-o",color="b",markersize=1)
        plt.plot(self.start.coord[0],self.start.coord[1],"xr")
        plt.plot(self.target.coord[0],self.target.coord[1],"xg")
        plt.show()
