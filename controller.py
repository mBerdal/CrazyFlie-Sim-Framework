from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import rot_matrix_zyx, ssa

class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data):
        pass

    @abstractmethod
    def update_set_point(self,set_point):
        pass

class DroneControllerPosPI(Controller):
    def __init__(self,pos_d: np.ndarray,**kwargs):
        self.pos_d = pos_d
        self.kp = kwargs.get("kp",0.7)
        self.ki = kwargs.get("ki",0.01)
        self.error_int = np.zeros((3,1))
        self.max_velocity = np.array([2,2,2,0.4,0.4,0.4]).reshape(6,1)
        self.timestep = 0.05

    def get_commands(self, states, sensor_data):
        Rnb = rot_matrix_zyx(states.item(3), states.item(4), states.item(5))

        pos_error = self.pos_d - states[0:3,:]
        pos_error_bdy = Rnb @ pos_error
        self.error_int += self.timestep*pos_error_bdy

        u_p = self.kp*pos_error_bdy
        u_i = self.ki*self.error_int
        u_sat = np.clip(u_p+u_i,-self.max_velocity[0:3],self.max_velocity[0:3])
        u_unsat = u_p + u_i
        self.error_int = self.error_int + self.timestep/self.ki*(u_sat-u_unsat)
        command = np.concatenate([u_sat,np.zeros([3,1])]).reshape(6,1)
        return command

    def update_set_point(self,set_point):
        self.pos_d = set_point
        self.error_int = np.zeros((3,1))

class DroneControllerPosHeading(Controller):

    def __init__(self,pos_d: np.ndarray, time_step=0.05,**kwargs):
        self.set_point = pos_d
        self.kp = kwargs.get("kp",0.7)
        self.kd = kwargs.get("kd",0.0)
        self.time_step = time_step
        self.heading_controller = HeadingControllerPD(0, time_step=time_step)
        self.velocity_controller = VelocityControllerPID(0, time_step=time_step)
        self.sat = kwargs.get("sat",{"heading": np.inf, "velocity": 1.5})
        self.velocity_gain = kwargs.get("velocity_gain", 1.5)

    def get_commands(self, states, sensor_data):
        err_x = self.set_point[0] - states[0]
        err_y = self.set_point[1] - states[1]

        dist_err = np.sqrt(err_x**2 + err_y**2)
        velocity_d = np.clip(self.velocity_gain*dist_err, -self.sat["velocity"], self.sat["velocity"])
        heading_d = np.arctan2(err_y,err_x) % (2*np.pi)

        self.heading_controller.update_set_point(heading_d)
        r_output = self.heading_controller.get_control_action(states[5])

        self.velocity_controller.update_set_point(velocity_d)
        u_output = self.velocity_controller.get_control_action(states[6])

        command = np.array([u_output, 0, 0, 0, 0, r_output],dtype=np.float).reshape(6, 1)
        return command

    def update_set_point(self, set_point):
        self.set_point = set_point


class WaypointController(Controller):

    def __init__(self, waypoints, **kwargs):
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.pos_controller = DroneControllerPosHeading(waypoints[self.current_waypoint])
        self.margin = kwargs.get("margin", 0.50)

    def get_commands(self, states, sensor_data):
        if np.linalg.norm(states[[0,1]]-self.waypoints[self.current_waypoint][0:2]) < self.margin:
            if self.current_waypoint < len(self.waypoints)-1:
                self.current_waypoint += 1
                self.pos_controller.update_set_point(self.waypoints[self.current_waypoint])
        return self.pos_controller.get_commands(states,sensor_data)

    def update_set_point(self, set_point):
        self.waypoints = set_point

class SwarmController(Controller):
    def __init__(self, drones, set_points):
        assert len(drones) == len(set_points)
        controllers = {}
        for d, s in zip(drones, set_points):
            controllers[d.id] = WaypointController(s)
        self.controllers = controllers

    def get_commands(self, states, sensor_data):
        commands = {}
        for key in self.controllers.keys():
            commands[key] = self.controllers[key].get_commands(states[key],sensor_data[key])
        return commands

    def update_set_point(self, set_point):
        for s in set_point:
            self.controllers[s['id']].update_set_point(s['set_point'])
            self.controllers[s['id']].error_int = np.zeros((3,1))

class LowLevelController(ABC):

    @abstractmethod
    def get_control_action(self,meas):
        pass

    @abstractmethod
    def update_set_point(self,set_point):
        pass


class HeadingControllerPD(LowLevelController):
    def __init__(self,set_point, time_step=0.05, **kwargs):
        self.set_point = set_point
        self.kp = kwargs.get("kp",4)
        self.kd = kwargs.get("kd",2)
        self.err_last = 0
        self.sat = kwargs.get("saturation", np.pi)
        self.time_step = time_step

    def get_control_action(self, meas):
        err = ssa(self.set_point, meas)
        d_term = (err - self.err_last) / self.time_step

        output_p = self.kp*err
        output_d = self.kd*d_term

        output = output_p + output_d
        output = np.clip(output, -self.sat, self.sat)

        self.err_last = err

        return output

    def update_set_point(self,set_point):
        self.set_point = set_point

class VelocityControllerPID(LowLevelController):
    def __init__(self,set_point, time_step=0.05,**kwargs):
        self.set_point = set_point
        self.kp = kwargs.get("kp", 2)
        self.kd = kwargs.get("kd", 0.4)
        self.ki = kwargs.get("ki", 1)
        self.err_last = 0
        self.err_int = 0
        self.sat = kwargs.get("sat", 3)
        self.time_step = time_step
        self.anti_windup = kwargs.get("anti_windup",True)

    def get_control_action(self,meas):
        err = self.set_point - meas
        self.err_int += self.time_step*err

        diff = (err - self.err_last) / self.time_step
        self.err_last = err

        output_p = self.kp*err
        output_d = self.kd*diff
        output_i = self.ki*self.err_int

        output = output_p + output_i + output_d
        output_sat = np.clip(output, -self.sat, self.sat)

        if self.anti_windup:
            self.err_int += self.time_step / self.ki * (output_sat - output)

        return output_sat

    def update_set_point(self, set_point):
        self.set_point = set_point

    def reset_integrator(self):
        self.err_int = 0
