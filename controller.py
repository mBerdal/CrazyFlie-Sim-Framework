from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import ssa
from slam.slammer import Slammer
from slam.shared_map import SharedMap
from sensor.odometry_sensor import OdometrySensor
from logger.loggable import Loggable
from logger.log_entry import LogEntry
from planning.path_planning import AStar
from planning.coordinator import Coordinator
from planning.dynamic_window import DynamicWindow
import matplotlib.pyplot as plt

class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data, time_step):
        pass

    @abstractmethod
    def update_set_point(self, set_point):
        pass

class DroneController(Controller, Loggable):
    """
    Controller for a single drone during autonmous exploration.

    Takes in the desired waypoints and navigate towards the goal using the Dynamic Window approach to perform obstacle
    avoidance.
    """

    def __init__(self, id,**kwargs):
        self.id = id
        self.state = "idle"

        self.waypoints = []
        self.current_waypoint = 0
        self.accept_wp_dist = 1.0
        self.loop_closing_dist = 0.2

        self.heading_controller = HeadingControllerPD(0)
        self.velocity_controller = VelocityControllerPID(0)
        self.yaw_controller = YawController(0)
        self.avoid_controller = DynamicWindow()

        self.time_counter = 0
        self.radius = 5

        self.update_flag = False

        self.sat = {"heading": np.inf, "velocity": 1.5}
        self.velocity_gain = kwargs.get("velocity_gain", 1)

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
            self.yaw_controller.update_set_point(0)
        else:
            update = ((self.time_counter % 0.10) - ((self.time_counter - time_step) % 0.10) < 0)
            if update:
                distance_grid = slammer.get_dist_grid(np.ceil(2))
                x = np.zeros([5,1])
                x[0:3] = self.estimated_pose[:]
                x[3:] = states[[6,11]]
                output = self.avoid_controller.calc_control(x,wp,distance_grid)
                self.velocity_controller.update_set_point(output[0])
                self.yaw_controller.update_set_point(output[1])
                self.update_flag = False

        self.prev_odometry = odometry_data

        r_output = self.yaw_controller.get_control_action(states[11], time_step)
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

    def add_waypoints(self, waypoints, state="active"):
        for w in waypoints:
            self.waypoints.append(w)
        self.update_flag = True
        self.state = state

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
        dist_wp = np.linalg.norm(pose[0:2] - self.waypoints[self.current_waypoint][0:2])
        if self.state == "active":
            if dist_wp < self.accept_wp_dist:
                    if self.current_waypoint < len(self.waypoints) - 1:
                        self.current_waypoint += 1
                        self.toggle_update_flag(True)
                    else:
                        self.state = "idle"
        elif self.state == "loop_closing":
            if dist_wp < self.accept_wp_dist:
                if self.current_waypoint == len(self.waypoints)-1:
                    if dist_wp < self.loop_closing_dist:
                        self.state = "idle"
                else:
                    self.current_waypoint += 1
                    self.toggle_update_flag(True)

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


class SwarmExplorationController(Controller, Loggable):
    """
    Controller class for controlling a swarm of drones performing autonmous exploration and mapping.

    Corrdination is achieved using the Coordinator class. Updates the waypoints given to the local controllers.
    Performes replanning procedures when neccessary.
    """

    def __init__(self, drones, initial_poses, rays, **kwargs):
        self.ids = []
        self.simulation_callback = None
        self.controllers = {}
        self.slammers = {}
        assignment_method = kwargs.get("assignment_method", "optimized")
        self.planner = Coordinator(assignment_method=assignment_method)
        self.initial_poses = initial_poses

        self.num_particles=kwargs.get("num_particles", 5)
        self.calibration_time = kwargs.get("calibration_time", 2.0)
        self.replanning_interval = kwargs.get("replanning_interval", 0.5)

        for ind, d in enumerate(drones):
            self.ids.append(d.id)
            self.controllers[d.id] = DroneController(d.id)
            self.slammers[d.id] = Slammer(d.id, self.num_particles, rays, initial_pose=initial_poses[ind].copy())
        initial_poses = [initial_poses[0] for _ in initial_poses]

        self.shared_map = SharedMap(self.ids[0], self.get_all_maps(), initial_poses)
        self.time_counter = 0
        self.time_last_update_wps = 0
        self.wait_time_update_wps = kwargs.get("wait_time_update_wps", 2)
        self.padding_occ_grid = 4

        self.visualize = kwargs.get("visualize", True)
        self.visualize_interval = kwargs.get("visualize_interval", 1.0)
        if self.visualize:
            self.init_plot()

        self.t_dist = 6
        self.d_dist = 20
        self.loop_closure_interval = 5
        self.loop_closure_count = 0
        self.td_ratio = 1

    def get_commands(self, states, sensor_data, time_step):
        commands = {}
        self.time_counter += time_step

        #Update the local SLAM for each drone based on the information given from the simulator
        for key in self.ids:
            update_flag = self.slammers[key].update(sensor_data[key], time_step)
            self.controllers[key].toggle_update_flag(update_flag)

        update_cooldown = (self.time_counter-self.time_last_update_wps) >= self.wait_time_update_wps
        idle_drones = self.get_idle_drones() != []
        calibration_ended = self.time_counter >= self.calibration_time

        #Perform coordination if there are any idle drones and a ceratiain amount of time have passed since last update.
        if update_cooldown and idle_drones and calibration_ended :
            self.assign_frontiers()

        replan = ((self.time_counter % self.replanning_interval) - ((self.time_counter - time_step) % self.replanning_interval)) < 0

        if replan and calibration_ended:
            self.replanning()

        for key in self.ids:
            commands[key] = self.controllers[key].get_commands(self.slammers[key], states[key], sensor_data[key], time_step)

        update_plot = ((self.time_counter % self.visualize_interval) - (
                    (self.time_counter - time_step) % self.visualize_interval)) < 0

        if self.visualize and update_plot:
            self.update_plot()
        return commands

    def replanning(self):

        #Function for checking if the current plan for each drone is feasible based on the current global map and
        #performes replanning if the current path is found unfeasible
        self.shared_map.merge_map(self.get_all_maps())
        occ_grid_end_check = self.shared_map.get_occupancy_grid(pad=2)
        occ_grid_planning = self.shared_map.get_occupancy_grid(pad=self.padding_occ_grid)

        for key in self.ids:
            if self.controllers[key].state == "idle":
                continue

            #Check if the final goal is reachable
            wp_end = self.controllers[key].get_end_wp()
            cell_wp_end = self.shared_map.cell_from_coordinate_local_map(key, wp_end)
            if occ_grid_end_check[cell_wp_end[0],cell_wp_end[1]] == -1:
                self.controllers[key].reset_waypoints()
                continue

            #Check if there is a collison along the line of sight between robot and current wp
            wp = self.controllers[key].get_assigned_wp()
            if wp is None:
                continue
            cell_drone = self.shared_map.cell_from_coordinate_local_map(key, self.slammers[key].get_pose())
            cell_wp = self.shared_map.cell_from_coordinate_local_map(key, wp)

            collision = False
            if occ_grid_end_check[cell_wp[0],cell_wp[1]] == -1:
                collision = True

            collision_check = self.shared_map.check_collision(cell_drone, cell_wp)

            collision = collision or collision_check

            if collision:
                #Perform replanning using A-Star on a padded occupancy grid
                print("Replanning ID:", key)
                target = self.controllers[key].get_end_wp()
                cell_target = self.shared_map.cell_from_coordinate_local_map(key, target)
                a = AStar(cell_drone, cell_target, occ_grid_planning)
                pad = self.padding_occ_grid - 1
                while not a.init_sucsess and pad >= 0:
                    a = AStar(cell_drone, cell_target, self.shared_map.get_occupancy_grid(pad=pad))
                    pad -= 1
                res = a.planning()
                if res is not None:
                    wp = [self.shared_map.coordinate_from_cell_local_map(key, w) for w in res[0]]
                    state = self.controllers[key].state
                    self.controllers[key].reset_waypoints()
                    self.controllers[key].add_waypoints(wp,state=state)
                else:
                    self.controllers[key].reset_waypoints()

    def assign_frontiers(self):
        #Find the best possible assignment to the drones given the current global map and poses.
        print("Assigning frontiers")
        self.time_last_update_wps = self.time_counter
        self.shared_map.merge_map(self.get_all_maps())
        wps = self.planner.assign_waypoints(self.get_drones(), self.shared_map, self.get_loop_closure())
        for i in wps.keys():
            wp = [self.shared_map.coordinate_from_cell_local_map(i, w) for w in wps[i][0]]
            self.controllers[i].reset_waypoints()
            self.controllers[i].add_waypoints(wp,state=wps[i][1])

        if self.check_all_drones_idle() and self.simulation_callback is not None:
            self.simulation_callback()

    def get_loop_closure(self):
        loops = {}
        for key in self.ids:
            loop = self.slammers[key].check_loop_closure(self.t_dist, self.d_dist)
            if not loop:
                continue
            loops[key] = loop
        return loops

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

class YawController(LowLevelController):
    def __init__(self, set_point, **kwargs):
        self.set_point = set_point
        self.kp = kwargs.get("kp", 4)
        self.kd = kwargs.get("kd", 0.4)
        self.err_last = 0
        self.err_int = 0
        self.sat = kwargs.get("sat", np.deg2rad(720))

    def get_control_action(self, meas, time_step):
        err = self.set_point - meas
        self.err_int += time_step * err

        diff = (err - self.err_last) / time_step
        self.err_last = err

        output_p = self.kp * err
        output_d = self.kd * diff
        output_i = self.ki * self.err_int

        output = output_p + output_i + output_d
        output_sat = np.clip(output, -self.sat, self.sat)

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
