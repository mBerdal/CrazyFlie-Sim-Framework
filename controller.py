from abc import ABC, abstractmethod
import numpy as np
from utils.rotation_utils import rot_matrix_zyx
class Controller(ABC):
    @abstractmethod
    def get_commands(self, states, sensor_data):
        pass

class DroneControllerPosPI(Controller):
    def __init__(self,pos_d: np.ndarray):
        self.pos_d = pos_d
        self.kp = 0.2

    def get_commands(self, states, sensor_data):
        Rnb = rot_matrix_zyx(states.item(3), states.item(4), states.item(5))

        pos_error = self.pos_d - states[1:3,:]
        pos_error_bdy = Rnb @ pos_error

        u_p = -self.kp*Rnb*pos_error_bdy

        command = np.concatenate([u_p,np.zeros([3,1])]).reshape(6,1)
        return command

    def update_set_point(self,set_point):
        self.pos_d = set_point


class SwarmController(Controller):
    def __init__(self, drones, set_points):
        assert len(drones) == len(set_points)
        controllers = {}
        for d,s in zip(drones,set_points):
            controllers[d.id] = DroneControllerPosPI(s)
        self.controllers = controllers

    def get_commands(self, states, sensor_data):
        commands = {}
        for key in self.controllers.keys():
            commands[key] = self.controllers[key].get_commands(states[key],sensor_data[key])
        return commands

    def update_set_points(self,set_points):
        for key in set_points:
            self.controllers[key].update_set_point(set_points[key])

