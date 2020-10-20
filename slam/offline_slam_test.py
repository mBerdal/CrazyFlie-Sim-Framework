import numpy as np
import matplotlib.pyplot as plt
from logger.logger import Logger
from sensor.lidar_sensor import LidarSensor
from matplotlib.animation import FuncAnimation
from slam.map import SLAM_map
from matplotlib import animation
from slam.grid_slam import Grid_SLAM
import matplotlib
from utils.rotation_utils import ssa
from scipy.stats import norm
#np.random.seed(1)

num_particles = 10

steps = 500
delta_step = 5

map_params = {
    "size_x": 400,
    "size_y": 400,
    "res": 0.1,
}

scan_match_params = {
    "max_iterations": 20,
    "step": 0.5,
    "kernel_size": 3,
    "kernel_var": 1
}

odometry_params = {
    "alpha1": 1,
    "alpha2": 1,
    "alpha3": 1,
    "alpha4": 1
}

obs_params = {
    "kernel_size": 3,
    "kernel_var": 1,
    "occ_threshold": 0.7,
    "max_range": 4
}
def test():
    log = Logger()
    log.load_from_file("slam/test_lidar.json")

    measurements = log.get_drone_sensor_measurements("0",0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0",0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy,sensor_specs.sensor_attitude_bdy,num_rays=72)
    rays = sensor.ray_vectors


    map = SLAM_map()
    initial_pose = states["states"][0][[0,1,5]]
    slam = Grid_SLAM(num_particles,initial_pose,rays,map_params=map_params,scan_match_params=scan_match_params,
                     obs_params=obs_params,odometry_params=odometry_params)
    maps = []
    for i in range(delta_step,steps,delta_step):
        print("Step:",i)
        meas = measurements["measurements"][i]
        prev = states["states"][i-delta_step][[0,1,5]]
        new = states["states"][i][[0,1,5]]
        odometry = np.concatenate([noise_odometry(prev,new),prev],axis=1).transpose()
        slam.update_particles(meas,odometry)
        print(slam.best_particle)
        maps.append({"map":slam.particles[slam.best_particle].map.convert_grid_to_prob(),"pose":slam.particles[slam.best_particle].pose.copy(),"time":i*states["step_length"]})



    def visualize():
        fig, axis = plt.subplots(1)
        drone = plt.Circle((states["states"][0][0] / map.res + map.center_x, states["states"][0][1] / map.res + map.center_y), radius=0.1,color="red")
        axis.add_patch(drone)

        def animate(t):
            axis.axis("equal")
            axis.set_title("Time: {:.2f}".format(maps[t]["time"]))
            axis.imshow(maps[t]["map"].transpose(),"Greys",origin="lower",extent=[-map.size_x/2*map.res,map.size_x/2*map.res,-map.size_x/2*map.res,map.size_x/2*map.res])
            drone.set_center([maps[t]["pose"][0], maps[t]["pose"][1]])
            print(maps[t]["pose"][2])
        anim = FuncAnimation(fig, animate, frames=[i for i in range(len(maps))], repeat=False, blit=False, interval=500)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #anim.save("map_building.mp4", writer=writer)
        plt.draw()
        plt.show()
    #visualize()


def noise_odometry(prev, new):
    #delta_rot= ssa(np.arctan2(new[1] - prev[1], new[0] - prev[0]), new[2])
    delta_rot = 0.1
    delta_trans_x = np.abs(new[0]-prev[0])+0.1
    delta_trans_y = np.abs(new[1]-prev[1])+0.1
    new[0] = new[0] + norm.rvs(scale=delta_trans_x**2)
    new[1] = new[1] + norm.rvs(scale=delta_trans_y**2)
    diff = norm.rvs(scale=delta_rot ** 2)
    new[2] = (new[2] + diff) % (2*np.pi)
    return new

if __name__ == "__main__":
    test()