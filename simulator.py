from environment import Environment
from crazy_flie import CrazyFlie
from range_sensor import RangeSensor
from communication import CommunicationChannel
from logger import Logger

import numpy as np
from enum import Enum



class Simulator():

  class SimulatorState(Enum):
    NORMAL = 0
    RECREATE = 1

  def __init__(self, environment: Environment, **kwargs) -> None:
    assert not environment is None, "environment must be a valid Environment object. Simulator constructor failed"
    assert "drones" in kwargs or "trajectories" in kwargs,\
      "Either drones = list of {'id': drone_id, 'state': state} and controller = Controller()\n\
      or trajectories = dict of {time_step: list[{drone_id, drone_state}]} must be supplied as kwargs.\n\
      Simulator constructor failed"

    self.environment = environment

    if "drones" in kwargs:
      assert "controller" in kwargs,\
        "controller must be supplied as a keyword argument when drone trajectories are not known. Simulator constructor failed"

      self.state = self.SimulatorState.NORMAL
      self.drones = {
        drone["id"]: CrazyFlie(
          drone["state"],
          [
            RangeSensor(
              4,
              0.001,
              np.deg2rad(27),
              np.deg2rad(27/9),
              np.array([0, 0, 0]).reshape(3, 1),
              np.array([0, 0, (np.pi/2)*(i + 1/2)]).reshape(3, 1)
            ) for i in range(4)
          ]
        ) for drone in kwargs["drones"]
      }

      self.com_channel = CommunicationChannel(
        lambda sender, recipient: kwargs["com_filter"](sender, recipient) if "com_filter" in kwargs else True,
        delay = kwargs["com_delay"] if "com_delay" in kwargs else None,
        packet_loss = kwargs["com_packet_loss"] if "com_packet_loss" in kwargs else None
      )
      # TODO: create controller class and assign
      # self.controller = controller
      self.logger = Logger(self.environment) if "log_sim" in kwargs and kwargs["log_sim"] else None
  
    elif "trajectories" in kwargs:
      self.state = self.SimulatorState.RECREATE
      self.drones = {
        drone["id"]: CrazyFlie(
          drone["state"],
          [
            RangeSensor(
              4,
              0.001,
              np.deg2rad(27),
              np.deg2rad(27/9),
              np.array([0, 0, 0]).reshape(3, 1),
              np.array([0, 0, (np.pi/2)*(i + 1/2)]).reshape(3, 1)
            ) for i in range(4)
          ]
        ) for drone in kwargs["trajectories"]["0.0"]
      }