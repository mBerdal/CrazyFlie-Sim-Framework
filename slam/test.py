import numpy as np
from math import sin, cos
import time
import matplotlib.pyplot as plt
from logger.logger import Logger
from sensor.lidar_sensor import LidarSensor
from matplotlib.animation import FuncAnimation
from slam.map import SLAM_map
from slam.scan_matcher import ScanMatcher
from matplotlib import animation
import matplotlib
#matplotlib.use("Agg")
np.random.seed(0)
def test():
    file = "test_lidar.json"
    log = Logger()
    log.load_from_file("test_lidar.json")

    measurements = log.get_drone_sensor_measurements("0",0)
    states = log.get_drone_states("0")
    sensor_specs = log.get_drone_sensor_specs("0",0)

    sensor = LidarSensor(sensor_specs.sensor_pos_bdy,sensor_specs.sensor_attitude_bdy,num_rays=144)
    rays = sensor.ray_vectors

    map = SLAM_map(size_x=400,size_y=400)
    scan = ScanMatcher(step=1)
    maps = []
    matched_diff = 0
    initial_diff = 0
    for i in range(0,states["steps"],5):
        meas = measurements["measurements"][i]
        pose = states["states"][i][[0,1,5]]
        map.integrate_scan(pose,meas,rays)
        if i % 20 == 0:
            initial_pose = pose + np.random.uniform(-0.2,0.2,pose.shape)
            matched_pose = scan.scan_match(rays,meas,initial_pose,map)
            matched_diff += np.linalg.norm(matched_pose - pose)
            initial_diff += np.linalg.norm(initial_pose - pose)
            print("\n Time:",i*states["step_length"])
            print("Initial diff:",initial_pose-pose)
            print("Mathced diff:", matched_pose-pose)
            print("Absolute initial diff:", np.linalg.norm(initial_pose-pose))
            print("Absolute matched diff:", np.linalg.norm(matched_pose - pose))
            maps.append({"map":map.convert_grid_to_prob(),"pose":pose,"time":i*states["step_length"],"matched_pose":matched_pose,"initial_pose":initial_pose})
    print("Matched diff:", matched_diff)
    print("Initial diff:", initial_diff)
    def visualize():
        fig, axis = plt.subplots(1)
        drone = plt.Circle((states["states"][0][0] / map.res + map.center_x, states["states"][0][1] / map.res + map.center_y), radius=0.5,color="red")
        drone_match = plt.Circle(
            (states["states"][0][0] / map.res + map.center_x, states["states"][0][1] / map.res + map.center_y),
            radius=0.5, color="blue")
        drone_initial = plt.Circle(
            (states["states"][0][0] / map.res + map.center_x, states["states"][0][1] / map.res + map.center_y),
            radius=0.5, color="green")
        axis.add_patch(drone)
        axis.add_patch(drone_match)
        axis.add_patch(drone_initial)

        def animate(t):
            axis.axis("equal")
            axis.set_title("Time: {:.2f}".format(t["time"]))
            axis.imshow(t["map"].transpose(),"Greys",origin="lower")
            drone.set_center([t["pose"][0] / map.res + map.center_x, t["pose"][1] / map.res + map.center_y])
            drone_match.set_center([t["matched_pose"][0] / map.res + map.center_x, t["matched_pose"][1] / map.res + map.center_y])
            drone_initial.set_center([t["initial_pose"][0] / map.res + map.center_x, t["initial_pose"][1] / map.res + map.center_y])
        anim = FuncAnimation(fig, animate, frames=maps, repeat=False, blit=False, interval=500)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        #anim.save("map_building.mp4", writer=writer)
        plt.draw()
        plt.show()
    visualize()

if __name__ == "__main__":
    test()