from logger.read_log_maps import read_slam_log, read_shared_map_log
import matplotlib.pyplot as plt
import numpy as np


def plot_last_maps(slam_file, shared_map_file=None, plot_trajectories=False):

    slam = read_slam_log(slam_file)
    ids = [i for i in slam.keys()]

    for i in ids:
        plt.figure(i)
        t = slam[i]["t_maps"]
        map = slam[i]["maps"][t[-1]]
        res = slam[i]["info"]["res"]
        size_x = slam[i]["info"]["size_x"]
        size_y = slam[i]["info"]["size_y"]
        extent = [-size_x/2*res, size_x/2*res,-size_y/2*res, size_y/2*res]
        plt.imshow(map.T, "Greys", origin="lower",extent=extent)
        if plot_trajectories:
            t_poses = slam[i]["t_poses"]
            ind = t_poses.index(t[-1])
            t_poses = t_poses[0:ind]

            t_x = [slam[i]["poses"][j][0] for j in t_poses]
            t_y = [slam[i]["poses"][j][1] for j in t_poses]

            plt.plot(t_x, t_y)
    if shared_map_file is not None:
        shared_map = read_shared_map_log(shared_map_file)
        plt.figure("SM")
        t = shared_map["timesteps"]
        map = shared_map["maps"][t[-1]]
        map = np.exp(map)/(1+np.exp(map))
        res = shared_map["info"]["res"]
        size_x = shared_map["info"]["size_x"]
        size_y = shared_map["info"]["size_y"]
        extent = [-size_x / 2 * res, size_x / 2 * res, -size_y / 2 * res, size_y / 2 * res]
        plt.imshow(map.T,"Greys", origin="lower", extent=extent)

    plt.show()


if __name__ == "__main__":
    slam_name = "logs/slam5.json"
    shared_name = "logs/shared_map5.json"
    plot_last_maps(slam_name, shared_map_file=shared_name, plot_trajectories=True)