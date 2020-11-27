import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

class RRT:
    class Node:
        def __init__(self, coord):
            self.coord = coord
            self.parent = None
            self.path_x = []
            self.path_y = []

    def __init__(self, start, target, map, free_cells, expand_dist=15, max_iter=500, target_sample_rate=0.20):
        self.nodes = []
        self.start = self.Node(np.array(start,np.int).reshape(2,1))
        self.target = self.Node(np.array(target,np.int).reshape(2,1))

        self.expand_dist = expand_dist
        self.target_sample_rate = target_sample_rate
        self.max_iter = max_iter
        self.map = map
        self.nodes_list = [self.start]

        self.free_cells = free_cells
        self.num_free_cells = len(self.free_cells)

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
            i = np.random.random_integers(0, self.num_free_cells-1)
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

        while np.any(current_cell != end.coord):
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
        if self.target.parent is None:
            return None
        wps = [self.target]
        current_node = self.target.parent
        prev_node = self.target
        i = 0
        c = 0
        while current_node is not self.start:
            if not self.check_collision(wps[i], current_node) and c < 5:
                prev_node = current_node
                current_node = current_node.parent
                c += 1
            else:
                wps.append(prev_node)
                i += 1
                c = 0
            if i > 100:
                return None

        wps = [w.coord for w in wps]
        wps.reverse()
        dist_list = [self.start.coord]
        for wp in wps:
            dist_list.append(wp)
        dist = sum([np.linalg.norm(dist_list[i] - dist_list[i+1]) for i in range(len(dist_list)-1)])
        return wps, dist

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
        plt.pause(0.1)


class RRTStar(RRT):
    class Node:
        def __init__(self, coord):
            self.coord = coord
            self.parent = None
            self.path_x = []
            self.path_y = []
            self.cost = 0

    def __init__(self, start, target, map, free_cells, run_to_max_iter=False, max_iter=500, expand_dist=15,
                 target_sample_rate=0.20):
        super().__init__(start, target, map, free_cells, max_iter=max_iter, expand_dist=expand_dist,
                         target_sample_rate=target_sample_rate)
        self.run_to_max_iter = run_to_max_iter

    def planning(self):
        iter = 0

        while iter < self.max_iter:
            rnd_node = self.get_random_node()
            ind_nearest = self.find_nearest_node(rnd_node)
            nearest_node = self.nodes_list[ind_nearest]

            node = self.steer(nearest_node, rnd_node)

            if node is not None:
                node.cost = nearest_node.cost + np.linalg.norm(nearest_node.coord - node.coord)
                near_inds = self.find_nearest_nodes(node)
                node_update_parent = self.choose_parent(node, near_inds)


                if node_update_parent:
                    self.rewire(node_update_parent, near_inds)
                    self.nodes_list.append(node_update_parent)
                    if not self.run_to_max_iter and np.linalg.norm(
                            node_update_parent.coord - self.target.coord) < self.expand_dist:
                        self.target.parent = node_update_parent
                        return self.get_path_to_target()
                else:
                    node.parent = nearest_node
                    self.nodes_list.append(node)

            iter += 1

        i = self.search_best_goal_node()
        if i is None:
            return None
        self.target.parent = self.nodes_list[i]
        return self.get_path_to_target()

    def find_nearest_nodes(self, new_node):
        dist_list = [np.linalg.norm(n.coord-new_node.coord) for n in self.nodes_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= self.expand_dist]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return None

        costs = []

        for i in near_inds:
            near_node = self.nodes_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node:
                costs.append(near_node.cost + np.linalg.norm(new_node.coord-near_node.coord))
            else:
                costs.append(np.inf)
        min_cost = min(costs)

        if min_cost == np.inf:
            return None

        min_ind = near_inds[costs.index(min_cost)]
        #new_node = self.steer(self.nodes_list[min_ind], new_node)
        new_node.cost = min_cost
        new_node.parent = self.nodes_list[min_ind]
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.nodes_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue

            edge_node.cost = new_node.cost + np.linalg.norm(new_node.coord-near_node.coord)

            if near_node.cost > edge_node.cost:
                near_node.coord = edge_node.coord
                near_node.cost = edge_node.cost
                near_node.parent = new_node
                self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.nodes_list:
            if node.parent == parent_node:
                node.cost = parent_node.cost + np.linalg.norm(parent_node.coord-node.coord)
                self.propagate_cost_to_leaves(node)

    def search_best_goal_node(self):
        dist_to_goal = [np.linalg.norm(n.coord - self.target.coord) for n in self.nodes_list]

        goal_inds = [dist_to_goal.index(i) for i in dist_to_goal if i <= self.expand_dist]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            if not self.check_collision(self.nodes_list[goal_ind], self.target):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.nodes_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.nodes_list[i].cost == min_cost:
                return i
        return None


class AStar:
    def __init__(self, start, target, occ_grid, movement=8):
        self.occ_grid = occ_grid
        start_x = np.int(np.floor(start[0]))
        start_y = np.int(np.floor(start[1]))

        self.start = (start_x,start_y)
        target_x = np.int(np.floor(target[0]))
        target_y = np.int(np.floor(target[1]))
        self.target = np.array([target_x,target_y])
        self.target = (target_x,target_y)
        self.init_sucsess = True
        if self.occ_grid[self.start[0],self.start[1]] == -1:
            self.init_sucsess = False

        if self.occ_grid[self.target[0],self.target[1]] == -1:
            self.init_sucsess = False

        if movement == 8:
            self.movements = self.get_movement_8()
        else:
            self.movements = self.get_movement_4()

    def planning(self):
        start_node = (self.distance(self.start, self.target), 0.0, self.start, None)

        front = [start_node]
        visited = np.zeros(self.occ_grid.shape)
        n = self.occ_grid.shape[0]
        m = self.occ_grid.shape[1]
        came_from = {}
        while front:
            elem = heappop(front)

            total_cost, cost, pos, previous = elem
            if visited[pos[0],pos[1]]:
                continue

            visited[pos[0],pos[1]] = 1

            came_from[(pos[0],pos[1])] = previous
            if np.all(pos == self.target):
                break

            for dx, dy, deltacost in self.movements:
                new_x = pos[0] + dx
                new_y = pos[1] + dy
                new_pos = (new_x, new_y)

                if new_x < 0 or new_x >= n or new_y < 0 or new_y >= m:
                    continue

                if not visited[new_pos[0],new_pos[1]] and not self.occ_grid[new_pos[0],new_pos[1]] == -1:
                    new_cost = cost + deltacost
                    new_estimated_cost_to_goal = new_cost + self.distance(new_pos, self.target)
                    heappush(front,(new_estimated_cost_to_goal,new_cost,new_pos,pos))

        path = []
        if np.all(pos == self.target):
            while pos is not None:
                path.append(pos)
                pos = came_from[(pos[0],pos[1])]
            path = [np.array([x[0],x[1]]).reshape(2,1) for x in path]
            path = self.get_path_to_target(path)
            return path
        else:
            return None

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

    def get_path_to_target(self, path):
        wps = [path[0]]
        current_node = path[1]
        prev_node = path[0]
        i = 0
        c = 2
        t = 0
        just_updated = False
        while c < len(path):
            if not self.check_collision(wps[i], current_node) and t < 40:
                prev_node = current_node
                current_node = path[c]
                just_updated = False
                c += 1
                t += 1
            else:
                if not just_updated:
                    wps.append(prev_node)
                    just_updated = True
                    i += 1
                    t = 0
                else:
                    dx = current_node[0]-prev_node[0]
                    dy = current_node[1]-prev_node[1]

                    if self.occ_grid[prev_node[0]+dx,prev_node[1]] != -1:
                        current_node[0] = prev_node[0]+dx
                        current_node[1] = prev_node[1]
                    elif self.occ_grid[prev_node[0],prev_node[1]+dy] != -1:
                        current_node[0] = prev_node[0]
                        current_node[1] = prev_node[1]+dy
        wps.reverse()
        dist_list = [np.array([self.start[0],self.start[1]]).reshape(2,1)]
        for wp in wps:
            dist_list.append(wp)
        dist = sum([np.linalg.norm(dist_list[i] - dist_list[i+1]) for i in range(len(dist_list)-1)])
        return wps, dist

    def distance(self, start, target):
        return np.sqrt((target[0]-start[0])**2 + (target[1]-start[1])**2)

    def get_movement_8(self):
        sqrt2 = np.sqrt(2)
        return [
            (1,0,1),
            (0,1,1),
            (-1,0,1),
            (0,-1,1),
            (1,1,sqrt2),
            (1,-1,sqrt2),
            (-1,1,sqrt2),
            (-1,-1,sqrt2)
        ]

    def get_movement_4(self):
        return [
            (1,0,1),
            (0,1,1),
            (-1,0,1),
            (0,-1,1)
        ]

    def visualize_visited(self, visited):
        plt.figure()
        plt.imshow(visited.T,"greys",origin="lower")
        plt.figure()
        plt.imshow(self.occ_grid.T,"greys",origin="lower")
        plt.show()