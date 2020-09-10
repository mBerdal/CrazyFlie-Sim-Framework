from environment import Environment
from crazy_flie import CrazyFlie
from range_sensor import RangeSensor
from communication import CommunicationChannel, CommunicationNode
from logger import Logger

import numpy as np
from enum import Enum

class Controller():
  def __init__(self):
    pass
  def get_commands(self, states, measurements):
    return 1


class Simulator(CommunicationNode):

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


      self.drones = Simulator.__load_drones(kwargs["drones"])
      self.drone_sensor_data = {}
      self.drone_states = {}
      self.commands = {d.id: 0 for d in self.drones}

      self.controller = kwargs["controller"]

      self.com_channel = CommunicationChannel(
        lambda sender, recipient: kwargs["com_filter"](sender, recipient) if "com_filter" in kwargs else True,
        delay = kwargs["com_delay"] if "com_delay" in kwargs else None,
        packet_loss = kwargs["com_packet_loss"] if "com_packet_loss" in kwargs else None
      )

      self.logger = Logger(self.environment) if "log_sim" in kwargs and kwargs["log_sim"] else None
  
    elif "trajectories" in kwargs:
      self.state = self.SimulatorState.RECREATE
      self.drones = Simulator.__load_drones(kwargs["trajectories"]["0.0"])

  @staticmethod
  def __load_drones(drone_list):
    return {
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
        ) for drone in drone_list
      }
  
  def revc_msg(self, msg_callback):
    msg_callback()
  
  def sim_step(self, time_step):
    
    def set_sensor_data_for_drone(drone_id, sensor_data):
      self.drone_sensor_data[drone_id] = sensor_data

    def set_state_for_drone(drone_id, drone_state):
      self.drone_states[drone_id] = drone_state

    msg_threads = []
    for d in self.drones:
      d.update_state(self.commands[d.id])
      sensor_data_thread = self.com_channel.send_msg(d, [self], set_sensor_data_for_drone(d.id, d.get_reading(self.environment)))
      drone_state_thread = self.com_channel.send_msg(d, [self], set_state_for_drone(d.id, d.state))
      msg_threads.append(sensor_data_thread, drone_state_thread)
    
    for t in msg_threads:
      t.join()

    self.commands = self.controller.get_commands(self.drone_states, self.drone_sensor_data)

    
