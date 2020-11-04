import numpy as np
import matplotlib.pyplot as plt
from logger.logger import Logger
from sensor.lidar_sensor import LidarSensor
from matplotlib.animation import FuncAnimation
from slam.map import SLAM_map
from matplotlib import animation
from slam.gridslam import GridSLAM
from utils.rotation_utils import ssa
from scipy.stats import norm
#np.random.seed(1)

file = "test_lidar_new.json"
num_particles = 5
drone_ids = ["0"]
steps = 500


slam_params = {
    "threshold_resampling": 1.5
}

particle_params = {
    "num_samples": 40,
    "eps": np.array([0.10,0.10,0.025],np.float).reshape(3,1)
}

map_params = {
    "size_x": 500,
    "size_y": 500,
    "res": 0.10,
    "max_range": 10,
}

scan_match_params = {
        "max_iterations": 5,
        "delta_x": 0.1,
        "delta_y": 0.1,
        "delta_theta": 0.05,
        "step": 1,
        "sigma": 0.1,
        "max_sensor_range": 10
    }

odometry_params = {
    "alpha1": 0.15,
    "alpha2": 0.10,
    "alpha3": 0.25,
    "alpha4": 0.25
}

obs_params = {
    "sigma": 0.15,
    "occ_threshold": 0.7,
    "max_sensor_range": 10
}

def test_multi_slam(file, drone_ids,num_particles,steps,slam_params, map_params, odometry_params, scan_params, obs_parms, particle_params):
    log = Logger()
    log.load_from_file(file)
    drone_info = {}
    slams = {}
    prev_state = {}
    prev_time = {}
    initial_pose = np.zeros([3, 1])
    slam_log = {}
    for id in drone_ids:
        info = {}
        info["measurements"] = log.get_drone_sensor_measurements(id,0)["measurements"]
        info["states"] = log.get_drone_states(id)["states"]
        info["sensor_specs"] = log.get_drone_sensor_specs(id,0)
        s = LidarSensor(info["sensor_specs"].sensor_pos_bdy, info["sensor_specs"].sensor_attitude_bdy, num_rays=144)
        rays = s.ray_vectors
        drone_info[id] = info
        slams[id] = GridSLAM(0,num_particles, initial_pose, rays, map_params=map_params, scan_match_params=scan_params,
                             obs_params=obs_params, odometry_params=odometry_params, particle_params=particle_params, **slam_params)
        prev_state[id] = info["states"][0][[0,1,5]]
        prev_time[id] = 0
        slam_log[id] = list()

    initial_pose_1 = np.array([2,2,0])
    initial_pose_2 = np.array([3,2,0])

    poses = [initial_pose_1,initial_pose_2]
    maps = [slams[id].particles[0].map for id in drone_ids]
    common_map = MapMultiRobot(maps, poses)
    plot_init = False
    for i in range(steps):
        print("Step:", i)
        for id in drone_ids:
            current_state = drone_info[id]["states"][i][[0,1,5]]
            update = False
            if i == 0:
                update = True
            if np.linalg.norm(prev_state[id][0:2] - current_state[0:2]) > 0.5:
                update = True
            if np.abs(ssa(prev_state[id][2], current_state[2])) > np.deg2rad(20):
                update = True
            if (i - prev_time[id]) > 10:
                update = True
            if update:
                odometry = np.concatenate([noise_odometry(prev_state[id], current_state), prev_state[id]], axis=1).transpose()
                meas = drone_info[id]["measurements"][i]
                slams[id].update_particles(meas, odometry)
                slam_log[id].append({"map": slams[id].particles[slams[id].best_particle].map.convert_grid_to_prob(),
                             "pose": slams[id].particles[slams[id].best_particle].pose.copy(), "time": i * 0.05})
                prev_state[id] = current_state
                prev_time[id] = i
        if i % 50 == 0 and i != 0:
            maps = [slams[id].get_best_map() for id in drone_ids]
            common_map.merge_map(maps)
            if not plot_init:
                fig_ids = drone_ids.copy()
                fig_ids.append("cm")
                figs = {}
                axs = {}
                for id in fig_ids:
                    fig = plt.figure(id)
                    ax = fig.add_subplot()
                    axs[id] = ax
                    figs[id] = fig

                #fig, axs = plt.subplots(nrows=(len(drone_ids)+1))
                objects = {}
                for id in drone_ids:
                    axs[id].set_title("Drone {:s}".format(id))
                    axs[id].set_xlabel("X [m]")
                    axs[id].set_ylabel("Y [m]")
                    ob = slams[id].init_plot(axs[id])
                    objects[id] = ob
                objects["cm"] = common_map.init_plot(axs["cm"])
                axs["cm"].set_title("Merged Map")
                axs["cm"].set_xlabel("X [m]")
                axs["cm"].set_ylabel("Y [m]")
                plot_init = True
            else:
                for id in fig_ids:
                    if id in drone_ids:
                        objects[id] = slams[id].update_plot(objects[id])
                    else:
                        objects[id] = common_map.update_plot(objects[id])
            for id in fig_ids:
                figs[id].canvas.draw()
            plt.pause(0.1)
    return slam_log

def visualize_slam(log, map_res, map_size_x,map_size_y):
    fig, axis = plt.subplots(1)
    start_state = log[0]["pose"]
    drone = plt.Circle((start_state[0] / map_res, start_state[1] / map_res), radius=0.1,color="red")
    axis.add_patch(drone)

    def animate(t):
        axis.axis("equal")
        axis.set_title("Time: {:.2f}".format(log[t]["time"]))
        axis.imshow(log[t]["map"].transpose(),"Greys",origin="lower",extent=[-map_size_x/2*map_res,map_size_x/2*map_res,-map_size_y/2*map_res,map_size_y/2*map_res])
        drone.set_center([log[t]["pose"][0], log[t]["pose"][1]])
        print(log[t]["pose"][2])
    anim = FuncAnimation(fig, animate, frames=[i for i in range(len(log))], repeat=False, blit=False, interval=500)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    #anim.save("map_building.mp4", writer=writer)
    plt.draw()
    plt.show()

def noise_odometry(prev, new):
    #delta_rot= ssa(np.arctan2(new[1] - prev[1], new[0] - prev[0]), new[2])
    delta_rot = ssa(new[2],prev[2])+0.01
    delta_trans_x = np.abs(new[0]-prev[0])+0.01
    delta_trans_y = np.abs(new[1]-prev[1])+0.01
    alpha1 = 0.5
    alpha2 = 0.5
    diff_x = norm.rvs(scale=alpha1*delta_trans_x**2)
    new[0] = new[0] + diff_x
    diff_y = norm.rvs(scale=alpha1*delta_trans_y**2)
    new[1] = new[1] + diff_y
    diff_rot = norm.rvs(scale=alpha2*delta_rot ** 2)
    new[2] = (new[2] + diff_rot) % (2*np.pi)
    return new




if __name__ == "__main__":
    log = test_multi_slam(file,drone_ids,num_particles,steps,slam_params,map_params,odometry_params,scan_match_params,obs_params,particle_params)
    for id in drone_ids:
        visualize_slam(log[id],map_params["res"],map_params["size_x"],map_params["size_y"])