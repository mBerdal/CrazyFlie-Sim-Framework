import numpy as np
from utils.rotation_utils import ssa
from queue import Queue


def gaussian_kernel_2d(size, sigma,res=0.1):
    xx = np.arange(-size,size+1,1)*res
    yy = np.arange(-size,size+1,1)*res
    x_grid, y_grid = np.meshgrid(xx,yy)
    kernel = np.exp(-((x_grid)**2 + (y_grid)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernel/np.max(kernel)


def gaussian_kernel_2d_v2(size, sigma,res=0.1):
    xx = np.arange(-size,size+1,1)
    yy = np.arange(-size,size+1,1)
    x_grid, y_grid = np.meshgrid(xx,yy)
    kernel = np.exp(-((x_grid*res)**2 + (y_grid*res)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernel


def create_straight_wall(corner1,corner2):
    delta_x = corner2[0]-corner1[0]
    delta_y = corner2[1]-corner1[1]
    delta_z = corner2[2]-corner1[2]
    x = corner1[0]
    y = corner1[1]
    z = corner1[2]
    obs1 = np.array([[x, x, x + delta_x, x + delta_x], [y, y, y, y], [z, z + delta_z, z, z + delta_z]])
    obs2 = np.array(
        [[x, x, x + delta_x, x + delta_x], [y+delta_y, y+delta_y, y + delta_y, y + delta_y], [z, z + delta_z, z, z + delta_z]])
    obs3 = np.array(
        [[x, x, x, x], [y, y, y + delta_y, y + delta_y], [z, z + delta_z, z, z + delta_z]])
    obs4 = np.array(
        [[x+delta_x, x+delta_x, x + delta_x, x + delta_x], [y, y, y + delta_y, y + delta_y], [z, z + delta_z, z, z + delta_z]])
    return [obs1,obs2,obs3,obs4]


def compute_entropy_map(map):
    map = np.clip(map, 0.00001, 0.99999)
    return -np.sum(map*np.log2(map) + (1-map)*np.log2(1-map))

def compute_information_map(map):
    map = np.clip(map,0.00001,0.99999)
    entropy = map*np.log2(map) + (1-map)*np.log2(1-map)
    return np.sum((1 + entropy))

def compute_mean_information_map(map):
    information = compute_information_map(map)
    n_obs = np.sum(map != 0.5)
    return information/n_obs



def compute_entropy_pose_discrete(poses, weights, res):
    cells = [(int(np.floor(p[0] /res)), int(np.floor(p[1] /res)),
              int(np.floor(p[2] / res))) for p in poses]
    unique = set(cells)
    entropy = 0
    for e in unique:
        indices = [i for i, x in enumerate(cells) if x == e]
        w = sum(weights[indices])
        entropy += - w * np.log(w)
    return entropy


def compute_entropy_pose_gaussian(poses, weights):
    mean = np.zeros([3, 1])
    for p, w in zip(poses,weights):
        mean[0:2] = p[0:2] * w
        mean[2] = ssa(p[2], 0) * w
    covariance = np.zeros([3, 3])
    for p, w in zip(poses,weights):
        diff = np.zeros([3, 1])
        diff[0:2] = p[0:2] - mean[0:2]
        diff[2] = ssa(p[2], mean[2])
        covariance += diff * diff.T * w
    return 0.5 * np.log(np.linalg.det(covariance) * (2 * np.pi * np.e) ** (3 / 2))


def compute_dist_loop_graph(loop_graph, current_node):
    q = Queue(0)
    q.put((current_node,0))
    visited = []
    dist = [0 for i in range(len(loop_graph))]
    while not q.empty():
        n, d = q.get()

        if n in visited:
            continue
        else:
            visited.append(n)
            dist[n] = d

        for e in loop_graph[n].edges:
            if e in visited:
                continue
            else:
                q.put((e,d+1))
    return dist

class Node:

    def __init__(self, pos, parent):
        self.pos = pos
        if parent is not None:
            self.edges = [parent]
        else:
            self.edges = []

def compute_frontiers(map, starting_cell, occ_threshold):
    cells = np.zeros(map.shape)
    cells[map > occ_threshold] = -1
    cells[map == 0] = 1

    q = Queue(0)
    q.put(starting_cell)

    visited = np.zeros(map.shape)

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

def test_loop_graph():
    graph = [Node(0, None), Node(1,0), Node(2,1), Node(3,2), Node(4,3), Node(5,4), Node(6,0)]
    graph[0].edges = [1, 5, 6]
    graph[1].edges = [0, 2]
    graph[2].edges = [1, 3]
    graph[3].edges = [2, 4]
    graph[4].edges = [3, 5]
    graph[5].edges = [4, 0]
    dist = compute_dist_loop_graph(graph,3)
    print(dist)

if __name__ == "__main__":
    test_loop_graph()