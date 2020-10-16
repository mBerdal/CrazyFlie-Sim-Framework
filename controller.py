from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import rot_matrix_zyx

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
        self.error_int += pos_error_bdy

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

class WaypointController(Controller):
    def __init__(self,waypoints):
        self.waypoints = waypoints
        self.current_waypoint = 0
        self.pos_controller = DroneControllerPosPI(waypoints[self.current_waypoint])
        self.diff = 0.2

    def get_commands(self, states, sensor_data):
        if np.linalg.norm(states[[0,1,5]]-self.waypoints[self.current_waypoint]) < self.diff:
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

