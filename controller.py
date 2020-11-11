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
from slam.path_planning import RRT, RRTStar
from slam.planner import Planner
import matplotlib.pyplot as plt

class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data, time_step):
        pass

    @abstractmethod
    def update_set_point(self, set_point):
        pass

class DroneController(Controller, Loggable):

    def __init__(self, id,**kwargs):
        self.id = id
        self.state = "idle"

        self.waypoints = []
        self.current_waypoint = 0
        self.accept_wp_dist = 0.5

        self.heading_controller = HeadingControllerPD(0)
        self.velocity_controller = VelocityControllerPID(0)
        self.avoid_controller = PredicativeCollisionAvoidance()

        self.time_counter = 0
        self.time_rate_collision_avoidance = kwargs.get("time_rate_collision_avoidance",0.5)
        self.radius = 5

        self.update_flag = False

        self.sat = {"heading": np.inf, "velocity": 1.5}
        self.velocity_gain = kwargs.get("velocity_gain", 0.5)

        self.estimated_pose = np.zeros([3,1])
        self.prev_odometry = np.zeros([3,1])
        self.output = np.zeros([6,1])

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

                self.avoid_controller.update_velocity_commands(np.array([0.25, 0.5, 1]) * velocity_d)

                heading_d, vel_d = self.avoid_controller.collision_avoidance(self.estimated_pose.copy(), velocity, distance_grid, wp)

                self.heading_controller.update_set_point((heading_d + self.estimated_pose[2]) % (2*np.pi))
                self.velocity_controller.update_set_point(vel_d)

                self.reset_time_counter()
                self.increment_timer(time_step)
                self.update_flag = False

        self.prev_odometry = odometry_data

        r_output = self.heading_controller.get_control_action(self.estimated_pose[2], time_step)
        u_output = self.velocity_controller.get_control_action(states[6], time_step)

        command = np.array([u_output, 0, 0, 0, 0, r_output], dtype=np.float).reshape(6, 1)
        self.output = command
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
        if self.waypoints == []:
            self.state = "idle"
            return None
        if np.linalg.norm(pose[0:2] - self.waypoints[self.current_waypoint][0:2]) < self.accept_wp_dist:
            if self.current_waypoint < len(self.waypoints) - 1:
                self.current_waypoint += 1
                self.toggle_update_flag(True)
            else:
                self.state = "idle"
        return self.waypoints[self.current_waypoint]

    def get_assigned_wp(self):
        try:
            return self.waypoints[self.current_waypoint]
        except IndexError:
            return None

    def get_end_wp(self):
        try:
            return self.waypoints[len(self.waypoints)-1]
        except IndexError:
            return None

    def generate_time_entry(self):
        return LogEntry(
            id=self.id,
            state=self.state,
            current_waypoint=self.current_waypoint,
            waypoints=self.waypoints,
            output=self.output,
            vel_controller=self.velocity_controller.generate_time_entry(),
            heading_controller=self.heading_controller.generate_time_entry()
        )

    def get_info_entry(self):
        return LogEntry(
            id=self.id,
            state=self.state,
            saturation=self.sat,
            velocity_gain=self.velocity_gain,
            vel_controller=self.velocity_controller.get_info_entry(),
            heading_controller=self.heading_controller.get_info_entry()
        )

    def get_time_entry(self):
        return LogEntry(
            id=self.id,
            state=self.state,
            current_waypoint=self.current_waypoint,
            waypoints=self.waypoints,
            output=self.output,
            vel_controller=self.velocity_controller.get_time_entry(),
            heading_controller=self.heading_controller.get_time_entry()
        )

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
        self.time_counter = 0
        self.update_map_time = 2


    def get_commands(self, states, sensor_data, time_step):
        commands = {}
        self.time_counter += time_step
        for key in self.controllers.keys():
            commands[key] = self.controllers[key].get_commands(states[key],sensor_data[key], time_step)
        return commands


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


class SwarmExplorationController(Controller, Loggable):

    def __init__(self, drones, initial_poses,**kwargs):
        self.ids = []
        self.simulation_callback = None
        self.controllers = {}
        self.slammers = {}
        self.planner = Planner()
        self.initial_poses = initial_poses
        rays = LidarSensor(np.array([0, 0, 0.1]), np.array([0, 0, 0]), num_rays=180).ray_vectors

        self.num_particles=kwargs.get("num_particles", 5)
        self.calibration_time = kwargs.get("calibration_time", 2.0)
        self.replanning_interval = kwargs.get("replanning_interval", 0.5)

        for d in drones:
            self.ids.append(d.id)
            self.controllers[d.id] = DroneController(d.id)
            self.slammers[d.id] = Slammer(d.id, self.num_particles, rays)

        self.shared_map = SharedMap(self.ids[0], self.get_all_maps(), initial_poses)
        self.time_counter = 0
        self.time_last_update_wps = 0
        self.wait_time_update_wps = 2

        self.visualize = kwargs.get("visualize",True)
        self.visualize_interval = kwargs.get("visualize_interval",2)
        if self.visualize:
            self.init_plot()

    def get_commands(self, states, sensor_data, time_step):
        commands = {}
        self.time_counter += time_step

        for key in self.ids:
            update_flag = self.slammers[key].update(sensor_data[key], time_step)
            self.controllers[key].toggle_update_flag(update_flag)

        update_cooldown = (self.time_counter-self.time_last_update_wps) >= self.wait_time_update_wps
        idle_drones = self.get_idle_drones() != []
        calibration_ended = self.time_counter >= self.calibration_time

        if update_cooldown and idle_drones and calibration_ended :
            print("Assigning frontiers")
            self.time_last_update_wps = self.time_counter
            self.shared_map.merge_map(self.get_all_maps())
            wps = self.planner.assign_waypoints_exploration(self.get_drones(),self.shared_map)
            for i in wps.keys():
                wp = [self.shared_map.coordinate_from_cell_local_map(i, w) for w in wps[i]]
                self.controllers[i].reset_waypoints()
                self.controllers[i].add_waypoints(wp)

            if self.check_all_drones_idle() and self.simulation_callback is not None:
                self.simulation_callback()

        replan = ((self.time_counter % self.replanning_interval) - ((self.time_counter - time_step) % self.replanning_interval)) < 0

        if replan and calibration_ended:
            print("Replanning")
            self.replanning()

        for key in self.ids:
            commands[key] = self.controllers[key].get_commands(self.slammers[key], states[key], sensor_data[key], time_step)

        update_plot = ((self.time_counter % self.visualize_interval) - (
                    (self.time_counter - time_step) % self.visualize_interval)) < 0

        if self.visualize and update_plot:
            self.update_plot()
        return commands

    def replanning(self):

        self.shared_map.merge_map(self.get_all_maps())
        occ_grid = self.shared_map.get_occupancy_grid(pad=2)
        free_cells = self.shared_map.get_free_cells()

        for key in self.ids:
            if self.controllers[key].state == "idle":
                continue

            wp = self.controllers[key].get_assigned_wp()
            if wp is None:
                continue

            cell_drone = self.shared_map.cell_from_coordinate_local_map(key, self.slammers[key].get_pose())
            cell_wp = self.shared_map.cell_from_coordinate_local_map(key, wp)
            collision = self.shared_map.check_collision(cell_drone, cell_wp)

            if collision:
                target = self.controllers[key].get_end_wp()
                cell_target = self.shared_map.cell_from_coordinate_local_map(key, target)
                r = RRTStar(cell_drone, cell_target, occ_grid, free_cells)
                res = r.planning()
                if res is not None:
                    wp = [self.shared_map.coordinate_from_cell_local_map(key, w) for w in res[0]]
                    self.controllers[key].reset_waypoints()
                    self.controllers[key].add_waypoints(wp)
                else:
                    self.controllers[key].reset_waypoints()

    def set_simulation_callback(self, callback):
        self.simulation_callback = callback

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
    def check_all_drones_idle(self):
        for i in self.ids:
            if self.controllers[i].state != "idle":
                return False
        return True

    def get_drones(self):
        drones = {}
        for d in self.ids:
            drones[d] = self.slammers[d].get_pose()
        return drones

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
            wp_end = self.controllers[i].get_end_wp()
            if wp is not None:
                wp = plt.Circle((wp[0], wp[1]), radius=0.1, color="blue")
                wp_end = plt.Circle((wp_end[0],wp_end[1]), radius=0.1,color="red")
            else:
                wp = plt.Circle((0,0), radius=0.1, color="blue")
                wp_end = plt.Circle((0, 0), radius=0.1, color="red")
            axs[i].add_patch(wp)
            axs[i].add_patch(wp_end)
            self.objects[i] = {"slammer": ob, "wp": wp, "wp_end": wp_end}

        self.objects["sm"] = self.shared_map.init_plot(axs["sm"])
        axs["sm"].set_title("Shared Map")
        axs["sm"].set_xlabel("X [m]")
        axs["sm"].set_ylabel("Y [m]")
        self.figs = figs
        plt.draw()
        plt.pause(0.01)

    def update_plot(self):
        for id in self.fig_ids:
            if id in self.ids:
                self.objects[id]["slammer"] = self.slammers[id].update_plot(self.objects[id]["slammer"])
                wp = self.controllers[id].get_assigned_wp()
                if wp is None:
                    self.objects[id]["wp"].set_center((0, 0))
                else:
                    self.objects[id]["wp"].set_center((wp[0],wp[1]))
                wp_end = self.controllers[id].get_end_wp()
                if wp_end is None:
                    self.objects[id]["wp_end"].set_center((0, 0))
                else:
                    self.objects[id]["wp_end"].set_center((wp_end[0],wp_end[1]))
            else:
                self.objects[id] = self.shared_map.update_plot(self.objects[id])
        for id in self.fig_ids:
            self.figs[id].canvas.draw()
        plt.pause(0.01)

    def get_info_entry(self):
        return LogEntry(
            ids=self.ids,
            controllers=[self.controllers[i].get_info_entry() for i in self.ids],
            slammers=[self.slammers[i].get_info_entry() for i in self.ids],
            shared_map=self.shared_map.get_info_entry()
        )

    def get_time_entry(self):
        return LogEntry(
            controllers=[self.controllers[i].get_time_entry() for i in self.ids],
            slammers=[self.slammers[i].get_time_entry() for i in self.ids],
            shared_map=self.shared_map.get_time_entry()
        )

    def generate_time_entry(self):
        return LogEntry(
            controllers=[self.controllers[i].generate_time_entry() for i in self.ids],
            slammers=[self.slammers[i].generate_time_entry() for i in self.ids],
            shared_map=self.shared_map.generate_time_entry()
        )
class LowLevelController(ABC):

    @abstractmethod
    def get_control_action(self,meas, time_step):
        pass

    @abstractmethod
    def update_set_point(self,set_point):
        pass


class HeadingControllerPD(LowLevelController, Loggable):
    def __init__(self,set_point, **kwargs):
        self.set_point = set_point
        self.kp = kwargs.get("kp",4)
        self.kd = kwargs.get("kd",3)
        self.err_last = 0
        self.sat = kwargs.get("saturation", np.pi*4)

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

    def get_info_entry(self):
        return LogEntry(
            kp=self.kp,
            kd=self.kd,
            saturation=self.sat
        )

    def get_time_entry(self):
        return LogEntry(
            set_point=self.set_point
        )

    def generate_time_entry(self):
        return LogEntry(
            set_point=self.set_point
        )


class VelocityControllerPID(LowLevelController, Loggable):
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

    def get_info_entry(self):
        return LogEntry(
            kp=self.kp,
            kd=self.kd,
            ki=self.ki,
            saturation=self.sat
        )

    def get_time_entry(self):
        return LogEntry(
            set_point=self.set_point
        )

    def generate_time_entry(self):
        return LogEntry(
            set_point=self.set_point
        )

