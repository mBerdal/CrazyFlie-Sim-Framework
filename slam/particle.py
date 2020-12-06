from slam.map import SLAM_map
from slam.scan_matcher import ScanMatcher
from slam.probability_models import OdometryModel, ObservationModel
from utils.rotation_utils import ssa
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.neighbors import NearestNeighbors

class Node:

    def __init__(self, pos, parent):
        self.pos = pos
        if parent is not None:
            self.edges = [parent]
        else:
            self.edges = []

    def __copy__(self):
        n = Node(self.pos.copy(), None)
        n.edges = self.edges.copy()
        return n

class Particle:

    def __init__(self, id, initial_weight, initial_pose, map_params={}, **kwargs):
        self.id = id
        self.weight = initial_weight
        self.pose = initial_pose

        self.map = SLAM_map(**map_params)

        self.trajectory = [initial_pose.copy()]
        self.counter = 0

        self.loop_graph = []
        self.current_node = 0
        self.total_nodes = 1
        self.loop_graph.append(Node(self.pose[0:2].copy(), None))
        self.min_dist_graph = 2.5
        self.dist_node_update = 0.25

    def update_weight(self, weight):
        self.weight = weight

    def update_loop_graph(self):
        dist = [np.linalg.norm(x.pos-self.pose[0:2]) for x in self.loop_graph]
        i = 0
        not_visible = True
        while not_visible and i < len(self.loop_graph):
            not_visible = not_visible and self.map.check_collision(self.pose[0:2], self.loop_graph[i].pos)
            i += 1
        if min(dist) > self.min_dist_graph or not_visible:
            self.loop_graph.append(Node(copy.deepcopy(self.pose[0:2]),self.current_node))
            self.loop_graph[self.current_node].edges.append(self.total_nodes)
            self.current_node = self.total_nodes
            self.total_nodes += 1
        elif min(dist) < self.dist_node_update and not not_visible:
            n = dist.index(min(dist))
            if self.current_node not in self.loop_graph[n].edges and n != self.current_node:
                self.loop_graph[n].edges.append(self.current_node)
                self.loop_graph[self.current_node].edges.append(n)
            self.current_node = n

    def __deepcopy__(self, memodict={}, **kwargs):
        par = Particle(self.id, copy.deepcopy(self.weight),copy.deepcopy(self.pose))
        par.map = self.map.__deepcopy__()
        par.trajectory = [p.copy() for p in self.trajectory]
        par.counter = self.counter
        par.loop_graph = [x.__copy__() for x in self.loop_graph].copy()
        par.current_node = self.current_node
        par.total_nodes = self.total_nodes
        return par

    def init_plot(self, axis):
        im = self.map.init_plot(axis)
        t,  = axis.plot([t[0] for t in self.trajectory], [t[1] for t in self.trajectory], "-o", color="green", markersize=1)
        d = plt.Circle((self.pose[0], self.pose[1]), radius=0.1, color="red")
        axis.add_patch(d)
        return {"drone": d, "map": im, "trajectory": t}

    def update_plot(self,objects):
        objects["map"] = self.map.update_plot(objects["map"])
        objects["trajectory"].set_xdata([t[0] for t in self.trajectory])
        objects["trajectory"].set_ydata([t[1] for t in self.trajectory])
        objects["drone"].set_center((self.pose[0], self.pose[1]))
        return objects

    def loop_graph_log(self):
        log = []
        for n in self.loop_graph:
            log.append((n.pos.tolist(),n.edges))
        return log

    def get_pose(self):
        return copy.deepcopy(self.pose)

    def visualize_loop_graph(self):
        plt.figure()

        plt.imshow(self.map.convert_grid_to_prob().transpose(), "Greys", origin="lower",
                   extent=[-self.map.size_x / 2 * self.map.res, self.map.size_x / 2 * self.map.res,
                           -self.map.size_y / 2 * self.map.res, self.map.size_y / 2 * self.map.res])
        plt.plot(self.pose[0], self.pose[1], "o", color="red", markersize=2)

        for n in self.loop_graph:
            for e in n.edges:
                plt.plot([n.pos[0],self.loop_graph[e].pos[0]], [n.pos[1],self.loop_graph[e].pos[1]], "-x", color="red", markersize=4)
        plt.pause(5)
        plt.close(plt.gcf())


    def visualize(self):
        plt.figure()

        plt.imshow(self.map.convert_grid_to_prob().transpose(),"Greys",origin="lower",
                   extent=[-self.map.size_x/2*self.map.res,self.map.size_x/2*self.map.res,-self.map.size_y/2*self.map.res,self.map.size_y/2*self.map.res])
        plt.plot(self.pose[0], self.pose[1], "o", color="red",markersize=2)
        plt.plot([t[0] for t in self.trajectory], [t[1] for t in self.trajectory], "-o", color="blue", markersize=2)

        plt.pause(0.1)

