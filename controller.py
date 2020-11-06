from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import ssa
from slam.slammer import Slammer
from slam.shared_map import SharedMap
from collision_avoidance import PredicativeCollisionAvoidance
from sensor.lidar_sensor import LidarSensor
from sensor.odometry_sensor import OdometrySensor
from logger.loggable import Loggable
from logger.log_entry import LogEntry
from slam.path_planning import RRT
import matplotlib.pyplot as plt

class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data, time_step):
        pass

    @abstractmethod
    def update_set_point(self, set_point):
        pass

class DroneSLAMController(Controller, Loggable):

    def __init__(self, id, pos_d, num_particles=5):
        self.id = id
        self.set_point = pos_d
        rays = LidarSensor(np.array([0,0,0.1]),np.array([0,0,0]),num_rays=180).ray_vectors
        self.slammer = Slammer(id,num_particles,rays)
        self.heading_controller = HeadingControllerPD(0)
        self.velocity_controller = VelocityControllerPID(0)
        self.avoidance_controller = PredicativeCollisionAvoidance()
        self.time_counter = 0
        self.time_rate_collision_avoidance = 1
        self.radius = 5
        self.update_flag = False
        self.sat = {"heading": np.inf, "velocity": 1.5}
        self.velocity_gain = 1.5
        self.estimated_heading = 0
        self.prev_odometry = np.zeros([3,1])

    def get_commands(self, states, sensor_data, time_step):
        slam_update = self.slammer.update(sensor_data,time_step)
        self.increment_timer(time_step)

        odometry_data = None
        for sd in sensor_data:
            if sd["type"] == OdometrySensor:
                odometry_data = sd["reading"]

        if self.time_counter >= self.time_rate_collision_avoidance or self.update_flag or slam_update:
            pose = self.slammer.get_pose()
            err_x = self.set_point[0] - pose[0]
            err_y = self.set_point[1] - pose[1]

            dist_err = np.sqrt(err_x ** 2 + err_y ** 2)
            velocity_d = np.clip(self.velocity_gain * dist_err, -self.sat["velocity"], self.sat["velocity"])

            distance_grid = self.slammer.get_occupancy_grid(np.ceil(velocity_d*5))
            velocity = states[6]

            self.avoidance_controller.update_velocity_commands(np.array([0.5,1])*velocity_d)

            heading_command, velocity_command = self.avoidance_controller.collision_avoidance(pose.copy(), velocity, distance_grid, self.set_point)

            self.heading_controller.update_set_point((heading_command + pose[2]) % (2*np.pi))
            self.velocity_controller.update_set_point(velocity_command)

            self.reset_time_counter()
            self.update_flag = False
            self.estimated_heading = pose[2]
        else:
            self.estimated_heading += ssa(odometry_data[2],self.prev_odometry[2])
            self.estimated_heading = self.estimated_heading %(2*np.pi)

        self.prev_odometry = odometry_data
        visualize = False
        if slam_update and visualize:
            self.slammer.visualize()
        r_output = self.heading_controller.get_control_action(self.estimated_heading, time_step)
        u_output = self.velocity_controller.get_control_action(states[6], time_step)
        command = np.array([u_output, 0, 0, 0, 0, r_output], dtype=np.float).reshape(6, 1)
        return command

    def update_set_point(self, set_point):
        self.set_point = set_point
        self.update_flag = True

    def increment_timer(self, time_step):
        self.time_counter += time_step

    def reset_time_counter(self):
        self.time_counter = 0

    def generate_time_entry(self):
        return self.slammer.generate_time_entry()

    def get_info_entry(self):
        return self.slammer.get_info_entry()

    def get_time_entry(self):
        return self.slammer.get_time_entry()

class DroneController(Controller, Loggable):

    def __init__(self, id):
        self.id = id
        self.state = "idle"

        self.waypoints = []
        self.current_waypoint = 0
        self.wp_dist = 1

        self.heading_controller = HeadingControllerPD(0)
        self.velocity_controller = VelocityControllerPID(0)
        self.avoid_controller = PredicativeCollisionAvoidance()

        self.time_counter = 0
        self.time_rate_collision_avoidance = 1
        self.radius = 5

        self.update_flag = False

        self.sat = {"heading": np.inf, "velocity": 1.5}
        self.velocity_gain = 1

        self.estimated_pose = np.zeros([3,1])
        self.prev_odometry = np.zeros([3,1])

    def get_commands(self, slammer, states, sensor_data, time_step):
        self.increment_timer(time_step)

        odometry_data = None
        for sd in sensor_data:
            if sd["type"] == OdometrySensor:
                odometry_data = sd["reading"]

        if self.update_flag:
            self.estimated_pose = slammer.get_pose()
        else:
            self.estimated_pose = self.odometry_update(odometry_data,self.prev_odometry)

        wp = self.get_waypoint(self.estimated_pose)

        if self.state == "idle":
            self.velocity_controller.update_set_point(0)

        elif self.state == "active":
            update_avoidance = (self.time_counter >= self.time_rate_collision_avoidance or self.update_flag)
            if update_avoidance:
                err_x = wp[0] - self.estimated_pose[0]
                err_y = wp[1] - self.estimated_pose[1]

                dist_err = np.sqrt(err_x ** 2 + err_y ** 2)
                velocity_d = np.clip(self.velocity_gain * dist_err, -self.sat["velocity"], self.sat["velocity"])

                distance_grid = slammer.get_occupancy_grid(np.ceil(velocity_d*5))
                velocity = states[6]

                self.avoid_controller.update_velocity_commands(np.array([0.5, 1]) * velocity_d)

                heading_d, vel_d = self.avoid_controller.collision_avoidance(self.estimated_pose.copy(), velocity, distance_grid, wp)

                self.heading_controller.update_set_point((heading_d + self.estimated_pose[2]) % (2*np.pi))
                self.velocity_controller.update_set_point(vel_d)

                self.reset_time_counter()
                self.update_flag = False

        self.prev_odometry = odometry_data

        r_output = self.heading_controller.get_control_action(self.estimated_pose[2], time_step)
        u_output = self.velocity_controller.get_control_action(states[6], time_step)

        command = np.array([u_output, 0, 0, 0, 0, r_output], dtype=np.float).reshape(6, 1)
        return command

    def odometry_update(self, current, prev):
        diff = np.zeros([3,1])
        diff[0:2] = (current[0:2] - prev[0:2]).reshape(2,1)
        diff[2] = ssa(current[2],prev[2])
        pose = np.zeros([3,1])

        pose[0] = self.estimated_pose[0] + diff[0]*np.cos(self.estimated_pose[2]) - diff[1]*np.sin(self.estimated_pose[2])
        pose[1] = self.estimated_pose[1] + diff[0]*np.sin(self.estimated_pose[2]) + diff[1]*np.cos(self.estimated_pose[2])
        pose[2] = (self.estimated_pose[2] + diff[2]) % (2*np.pi)
        return pose

    def add_waypoints(self, waypoints):
        for w in waypoints:
            self.waypoints.append(w)
        self.update_flag = True
        self.state = "active"

    def reset_waypoints(self):
        self.waypoints = []
        self.current_waypoint = 0
        self.state = "idle"

    def toggle_update_flag(self, flag):
        self.update_flag = (flag or self.update_flag)

    def increment_timer(self, time_step):
        self.time_counter += time_step

    def reset_time_counter(self):
        self.time_counter = 0

    def get_waypoint(self, pose):
        if np.linalg.norm(pose[0:2] - self.waypoints[self.current_waypoint][0:2]) < self.wp_dist:
            if self.current_waypoint < len(self.waypoints) - 1:
                self.current_waypoint += 1
            else:
                self.state = "idle"
        return self.waypoints[self.current_waypoint]

    def get_assigned_wp(self):
        return self.waypoints[self.current_waypoint]

    def get_end_wp(self):
        return self.waypoints[len(self.waypoints)-1]

    def generate_time_entry(self):
        pass

    def get_info_entry(self):
        pass

    def get_time_entry(self):
        pass

    def update_set_point(self, set_point):
        pass


class DroneControllerPosHeading(Controller, Loggable):

    def __init__(self,pos_d: np.ndarray,**kwargs):
        self.set_point = pos_d
        self.heading_controller = HeadingControllerPD(0)
        self.velocity_controller = VelocityControllerPID(0)
        self.sat = kwargs.get("sat", {"heading": np.inf, "velocity": 1.5})
        self.velocity_gain = kwargs.get("velocity_gain", 1.5)

    def get_commands(self, states, sensor_data, time_step):
        err_x = self.set_point[0] - states[0]
        err_y = self.set_point[1] - states[1]

        dist_err = np.sqrt(err_x**2 + err_y**2)
        velocity_d = np.clip(self.velocity_gain*dist_err, -self.sat["velocity"], self.sat["velocity"])
        heading_d = np.arctan2(err_y,err_x) % (2*np.pi)

        self.heading_controller.update_set_point(heading_d)
        r_output = self.heading_controller.get_control_action(states[5], time_step)

        self.velocity_controller.update_set_point(velocity_d)
        u_output = self.velocity_controller.get_control_action(states[6], time_step)

        command = np.array([u_output, 0, 0, 0, 0, r_output],dtype=np.float).reshape(6, 1)
        return command

    def update_set_point(self, set_point):
        self.set_point = set_point

    def generate_time_entry(self):
        return LogEntry(id=0)

    def get_info_entry(self):
        return LogEntry(id=0)

    def get_time_entry(self):
        return LogEntry(id=0)


class WaypointController(Controller, Loggable):

    def __init__(self, id, waypoints, **kwargs):
        self.id = id
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.pos_controller = DroneSLAMController(id, self.waypoints[self.current_waypoint])
        self.margin = kwargs.get("margin", 0.50)

    def get_commands(self, states, sensor_data, time_step):
        if np.linalg.norm(states[[0,1]]-self.waypoints[self.current_waypoint][0:2]) < self.margin:
            if self.current_waypoint < len(self.waypoints)-1:
                self.current_waypoint += 1
                self.pos_controller.update_set_point(self.waypoints[self.current_waypoint])
        return self.pos_controller.get_commands(states,sensor_data, time_step)

    def update_set_point(self, set_point):
        self.waypoints = set_point

    def get_info_entry(self):
        return self.pos_controller.get_info_entry()

    def get_time_entry(self):
        return self.pos_controller.get_time_entry()

    def generate_time_entry(self):
        return self.pos_controller.generate_time_entry()


class SwarmController(Controller,Loggable):
    def __init__(self, drones, set_points):
        assert len(drones) == len(set_points)
        controllers = {}
        poses = {}
        for d, s in zip(drones, set_points):
            controllers[d.id] = WaypointController(d.id,s)
            poses[d.id] = d.state[[0,1,5]]
        self.controllers = controllers
        self.map_merging = self.init_map_merging(poses)
        self.time_counter = 0
        self.update_map_time = 2


    def get_commands(self, states, sensor_data, time_step):
        commands = {}
        self.time_counter += time_step
        for key in self.controllers.keys():
            commands[key] = self.controllers[key].get_commands(states[key],sensor_data[key], time_step)
        if self.time_counter > self.update_map_time:
            maps = []
            for key in self.controllers.keys():
                maps.append(self.controllers[key].pos_controller.slammer.get_map())
            self.map_merging.merge_map(maps)
            #self.map_merging.visualize()
            frontiers = self.map_merging.compute_frontiers()
            self.map_merging.visualize_frontiers(frontiers)
            self.time_counter = 0
        return commands

    def init_map_merging(self, relative_poses):
        maps = []
        poses = []
        for key in self.controllers:
            poses.append(relative_poses[key])
            maps.append(self.controllers[key].pos_controller.slammer.get_map())
        return MapMultiRobot(maps,poses)


    def update_set_point(self, set_point):
        for s in set_point:
            self.controllers[s['id']].update_set_point(s['set_point'])

    def get_info_entry(self):
        return LogEntry(
            ids=self.controllers.keys(),
            controllers=[self.controllers[i].get_info_entry() for i in self.controllers.keys()]
        )

    def get_time_entry(self):
        return LogEntry(
            controllers=[self.controllers[i].get_time_entry() for i in self.controllers.keys()]
        )

    def generate_time_entry(self):
        return LogEntry(
            controllers=[self.controllers[i].generate_time_entry() for i in self.controllers.keys()]
        )


class SwarmExplorationController(Controller):

    def __init__(self, drones, initial_poses):
        self.ids = []
        self.controllers = {}
        self.slammers = {}
        self.initial_poses = initial_poses
        rays = LidarSensor(np.array([0, 0, 0.1]), np.array([0, 0, 0]), num_rays=180).ray_vectors
        for d in drones:
            self.ids.append(d.id)
            self.controllers[d.id] = DroneController(d.id)
            self.slammers[d.id] = Slammer(id, 5, rays)

        self.shared_map = SharedMap(self.ids[0], self.get_all_maps(), initial_poses)
        self.time_counter = 0
        self.update_rate_map = 3

        for k in self.ids:
            self.controllers[k].add_waypoints([np.array([1,0,0]).reshape(3,1)])
        self.init_plot()
        self.calibration_time = 2

    def get_commands(self, states, sensor_data, time_step):
        commands = {}
        self.time_counter += time_step

        for key in self.ids:
            update_flag = self.slammers[key].update(sensor_data[key],time_step)
            self.controllers[key].toggle_update_flag(update_flag)
            commands[key] = self.controllers[key].get_commands(self.slammers[key],states[key], sensor_data[key], time_step)

        if self.get_idle_drones() != [] and self.time_counter >= self.calibration_time:

            self.shared_map.merge_map(self.get_all_maps())

            idle_drones = self.get_idle_drones()
            active_drones = self.get_active_drones()

            idle_drones_pose = self.get_poses(idle_drones)
            active_drones_end_wp = self.get_assigned_wps(active_drones)

            idle_drones_info = {i: idle_drones_pose[i] for i in idle_drones}
            active_drones_info = {i: active_drones_end_wp[i] for i in active_drones}

            wps = self.shared_map.get_waypoint_exploration(idle_drones_info, active_drones_info)
            for ind, i in enumerate(idle_drones):
                start = self.shared_map.compute_cell(i,  self.slammers[i].get_pose())
                free_cells = self.shared_map.get_free_cells()
                c = 0
                while c < 20:
                    r = RRT(start, wps[ind]["point"], self.shared_map.get_occupancy_grid(), free_cells)
                    wp = r.planning()
                    if wp is not None:
                        break
                    c += 1

                if wp is None:
                    continue

                wp = [self.shared_map.compute_coordinate_local_map(i, w) for w in wp]
                self.controllers[i].reset_waypoints()
                self.controllers[i].add_waypoints(wp)

        self.update_plot()
        return commands

    def get_poses(self, ids):
        poses = {}
        for i in ids:
            poses[i] = self.slammers[i].get_pose()
        return poses

    def get_assigned_wps(self, ids):
        wps = {}
        for i in ids:
            wps[i] = self.controllers[i].get_end_wp()
        return wps

    def get_all_maps(self):
        maps = {}
        for key in self.ids:
            maps[key] = self.slammers[key].get_map()
        return maps

    def get_idle_drones(self):
        idle_drones = []
        for i in self.ids:
            if self.controllers[i].state == "idle":
                idle_drones.append(i)
        return idle_drones

    def get_active_drones(self):
        active_drones = []
        for i in self.ids:
            if self.controllers[i].state == "active":
                active_drones.append(i)
        return active_drones

    def update_set_point(self, set_point):
        pass

    def visualize(self):
        for i in self.ids:
            self.slammers[i].visualize()
        self.shared_map.visualize()

    def init_plot(self):
        self.fig_ids = self.ids.copy()
        self.fig_ids.append("sm")
        figs = {}
        axs = {}
        for i in self.fig_ids:
            fig = plt.figure(i)
            ax = fig.add_subplot()
            axs[i] = ax
            figs[i] = fig

        self.objects = {}
        for i in self.ids:
            axs[i].set_title("Drone {}".format(i))
            axs[i].set_xlabel("X [m]")
            axs[i].set_ylabel("Y [m]")
            ob = self.slammers[i].init_plot(axs[i])
            wp = self.controllers[i].get_assigned_wp()
            wp = plt.Circle((wp[0], wp[1]), radius=0.1, color="blue")
            axs[i].add_patch(wp)
            self.objects[i] = {"slammer": ob, "wp": wp}

        self.objects["sm"] = self.shared_map.init_plot(axs["sm"])
        axs["sm"].set_title("Shared Map")
        axs["sm"].set_xlabel("X [m]")
        axs["sm"].set_ylabel("Y [m]")
        self.figs = figs
        plt.draw()
        plt.pause(0.1)

    def update_plot(self):
        for id in self.fig_ids:
            if id in self.ids:
                self.objects[id]["slammer"] = self.slammers[id].update_plot(self.objects[id]["slammer"])
                wp = self.controllers[id].get_assigned_wp()
                self.objects[id]["wp"].set_center((wp[0],wp[1]))
            else:
                self.objects[id] = self.shared_map.update_plot(self.objects[id])
        for id in self.fig_ids:
            self.figs[id].canvas.draw()
        plt.pause(0.1)

class LowLevelController(ABC):

    @abstractmethod
    def get_control_action(self,meas, time_step):
        pass

    @abstractmethod
    def update_set_point(self,set_point):
        pass


class HeadingControllerPD(LowLevelController):
    def __init__(self,set_point, **kwargs):
        self.set_point = set_point
        self.kp = kwargs.get("kp",4)
        self.kd = kwargs.get("kd",3)
        self.err_last = 0
        self.sat = kwargs.get("saturation", np.pi/2)

    def get_control_action(self, meas, time_step):
        err = ssa(self.set_point, meas)
        d_term = (err - self.err_last) /time_step

        output_p = self.kp*err
        output_d = self.kd*d_term

        output = output_p + output_d
        output = np.clip(output, -self.sat, self.sat)

        self.err_last = err

        return output

    def update_set_point(self,set_point):
        self.set_point = set_point


class VelocityControllerPID(LowLevelController):
    def __init__(self,set_point,**kwargs):
        self.set_point = set_point
        self.kp = kwargs.get("kp", 2)
        self.kd = kwargs.get("kd", 0.4)
        self.ki = kwargs.get("ki", 1)
        self.err_last = 0
        self.err_int = 0
        self.sat = kwargs.get("sat", 3)
        self.anti_windup = kwargs.get("anti_windup",True)

    def get_control_action(self,meas, time_step):
        err = self.set_point - meas
        self.err_int += time_step*err

        diff = (err - self.err_last) / time_step
        self.err_last = err

        output_p = self.kp*err
        output_d = self.kd*diff
        output_i = self.ki*self.err_int

        output = output_p + output_i + output_d
        output_sat = np.clip(output, -self.sat, self.sat)

        if self.anti_windup:
            self.err_int += time_step / self.ki * (output_sat - output)

        return output_sat

    def update_set_point(self, set_point):
        self.set_point = set_point

    def reset_integrator(self):
        self.err_int = 0
