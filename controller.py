from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import ssa
from slam.slammer import Slammer
from collision_avoidance import PredicativeCollisionAvoidance
from sensor.lidar_sensor import LidarSensor
from sensor.odometry_sensor import OdometrySensor
from logger.loggable import Loggable
from logger.log_entry import LogEntry

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
        rays = LidarSensor(np.array([0,0,0.1]),np.array([0,0,0]),num_rays=144).ray_vectors
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
        visualize = True
        if slam_update and visualize:
            self.slammer.visualize()
            #self.heading_controller.update_set_point(heading_d)
            #self.velocity_controller.update_set_point(velocity_d)
        print("Estimated heading:",self.estimated_heading)
        print("Desired heading:",self.heading_controller.set_point)
        r_output = self.heading_controller.get_control_action(self.estimated_heading, time_step)
        u_output = self.velocity_controller.get_control_action(states[6], time_step)
        #print("Input yaw:",r_output.item())
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
        for d, s in zip(drones, set_points):
            controllers[d.id] = WaypointController(d.id,s)
        self.controllers = controllers

    def get_commands(self, states, sensor_data, time_step):
        commands = {}
        for key in self.controllers.keys():
            commands[key] = self.controllers[key].get_commands(states[key],sensor_data[key], time_step)
        return commands

    def update_set_point(self, set_point):
        for s in set_point:
            self.controllers[s['id']].update_set_point(s['set_point'])
            self.controllers[s['id']].error_int = np.zeros((3,1))

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
