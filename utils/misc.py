import numpy as np

def gaussian_kernel_2d(size, sigma,res=0.1):
    xx = np.arange(-size,size+1,1)
    yy = np.arange(-size,size+1,1)
    x_grid, y_grid = np.meshgrid(xx,yy)
    kernel = np.exp(-((res*x_grid)**2 + (res*y_grid)**2)/(2*sigma**2))/(2*np.pi*sigma**2)
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