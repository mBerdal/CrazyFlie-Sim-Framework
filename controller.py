from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import rot_matrix_zyx, ssa

class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data):
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

class DroneControllerPosPIHeading(Controller):
    def __init__(self,pos_d: np.ndarray,**kwargs):
        self.pos_d = pos_d
        self.kp = kwargs.get("kp",0.7)
        self.kd = kwargs.get("kd",0.0)
        self.timestep = 0.05
        self.heading_controller = HeadingControllerPD(0)
        self.sat = 2
        self.prev_dist = 0

    def get_commands(self, states, sensor_data):
        err_x = self.pos_d[0] - states[0]
        err_y = self.pos_d[1] - states[1]

        dist_err = np.sqrt(err_x**2 + err_y**2)
        heading_d = np.arctan2(err_y,err_x) % (2*np.pi)

        self.heading_controller.update_setpoint(heading_d)

        r_input = self.heading_controller.get_controll_action(states[5])
        dist_diff = (dist_err-self.prev_dist)/self.timestep

        u_input = np.clip(self.kp*dist_err+self.kd*dist_diff,-self.sat,self.sat)

        command = np.array([u_input,0,0,0,0,r_input]).reshape(6,1)
        return command
    def update_set_point(self, pos_d):
        self.pos_d = pos_d

class HeadingControllerPD():
    def __init__(self,heading_d):
        self.heading_d = heading_d
        self.kp = 2
        self.kd = 0.4
        self.error_prior = 0
        self.saturation = np.pi
        self.timestep = 0.05

    def get_controll_action(self, heading):
        err = ssa(self.heading_d,heading)

        diff = (err - self.error_prior)/self.timestep
        u_p = self.kp*err
        u_d = self.kd*diff

        u_unsat = u_p + u_d
        u_sat = np.clip(u_unsat,-self.saturation,self.saturation)
        self.error_prior = err
        return u_sat

    def update_setpoint(self, heading_d):
        self.heading_d = heading_d


class WaypointController(Controller):
    def __init__(self,waypoints):
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.pos_controller = DroneControllerPosPIHeading(waypoints[self.current_waypoint])
        self.diff = 0.25

    def get_commands(self, states, sensor_data):
        if np.linalg.norm(states[[0,1]]-self.waypoints[self.current_waypoint][0:2]) < self.diff:
            if self.current_waypoint < len(self.waypoints)-1:
                self.current_waypoint += 1
                self.pos_controller.update_set_point(self.waypoints[self.current_waypoint])
        return self.pos_controller.get_commands(states,sensor_data)


class SwarmController(Controller):
    def __init__(self, drones, set_points):
        assert len(drones) == len(set_points)
        controllers = {}
        for d,s in zip(drones,set_points):
            controllers[d.id] = WaypointController(s)
        self.controllers = controllers

    def get_commands(self, states, sensor_data):
        commands = {}
        for key in self.controllers.keys():
            commands[key] = self.controllers[key].get_commands(states[key],sensor_data[key])
        return commands

    def update_set_points(self,set_points):
        for s in set_points:
            self.controllers[s['id']].update_set_point(s['set_point'])
            self.controllers[s['id']].error_int = np.zeros((3,1))

