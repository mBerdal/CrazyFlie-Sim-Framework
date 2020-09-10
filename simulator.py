from environment import Environment
from crazy_flie import CrazyFlie
from range_sensor import RangeSensor
from communication import CommunicationNode, CommunicationChannel
from logger import Logger

import numpy as np
from enum import Enum

class Controller():
  def __init__(self):
    pass
  def get_commands(self, states, measurements):
    return np.zeros((6, 1))

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
      self.commands = {d.id: np.zeros((6, 1)) for d in self.drones}

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
    return [
        CrazyFlie(
          drone["id"],
          drone["state"]
        ) for drone in drone_list
    ]
  
  def recv_msg(self, msg):
    if not msg is None:
      msg()
  
  def sim_step(self, time_step):
    
    def set_sensor_data_for_drone(drone_id, sensor_data):
      self.drone_sensor_data[drone_id] = sensor_data

    def set_state_for_drone(drone_id, drone_state):
      self.drone_states[drone_id] = drone_state

    msg_threads = []
    for d in self.drones:
      sensor_data_thread = self.com_channel.send_msg(d, [self], set_sensor_data_for_drone(d.id, d.read_sensors(self.environment)))
      drone_state_thread = self.com_channel.send_msg(d, [self], set_state_for_drone(d.id, d.state))
      msg_threads.append(sensor_data_thread[0])
      msg_threads.append(drone_state_thread[0])
    
    for t in msg_threads:
      t.join()

    self.commands = self.controller.get_commands(self.drone_states, self.drone_sensor_data)
    for d in self.drones:
      d.update_command(self.commands[d.id])
      d.update_state(self.commands[d.id])


#def main():
#  e = Environment([
#    [True, False],
#    [True, False],
#  ], map_resolution=(1, 1))
#  c = Controller()
#  s = Simulator(e, drones = [{"id": 1, "state": np.zeros((6, 1))}], controller = c)
#  for _ in range(1000):
#    s.sim_step(0.10)
#
#main()
    
