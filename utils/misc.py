import numpy as np
from utils.rotation_utils import ssa

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
    map = np.where(map < 0.00001, 0.00001, map)
    map = np.where(map > 0.99999, 0.99999, map)
    return -np.sum(map*np.log(map) + (1-map)*np.log(1-map))

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
