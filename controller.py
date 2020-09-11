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
        self.kp = kwargs.get("kp",0.3)
        self.ki = kwargs.get("ki",0.001)
        self.error_int = np.zeros((3,1))

    def get_commands(self, states, sensor_data):
        Rnb = rot_matrix_zyx(states.item(3), states.item(4), states.item(5))

        pos_error = self.pos_d - states[0:3,:]
        pos_error_bdy = Rnb @ pos_error
        self.error_int += pos_error_bdy

        u_p = self.kp*pos_error_bdy
        u_i = self.ki*self.error_int
        command = np.concatenate([u_p+u_i,np.zeros([3,1])]).reshape(6,1)
        return command

    def update_set_point(self,set_point):
        self.pos_d = set_point


class SwarmController(Controller):
    def __init__(self, drones, set_points):
        assert len(drones) == len(set_points)
        controllers = {}
        for d,s in zip(drones,set_points):
            controllers[d['id']] = DroneControllerPosPI(s)
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

