import numpy as np

def gaussian_kernel_2d(size, sigma):
    xx = np.arange(-size,size+1,1)
    yy = np.arange(-size,size+1,1)
    x_grid, y_grid = np.meshgrid(xx,yy)
    kernel = np.exp(-(x_grid**2 + y_grid**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernel