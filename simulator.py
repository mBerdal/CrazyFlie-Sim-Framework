from environment.environment import Environment
from drone_swarm.drone.crazy_flie import CrazyFlie
from drone_swarm.drone_swarm import DroneSwarm
from communication import CommunicationNode, CommunicationChannel
from logger.logger import Logger
from utils.raytracing import multi_ray_intersect_triangle
import matplotlib.pyplot as plt

import numpy as np
from enum import Enum

class Simulator(CommunicationNode):

  class SimulatorState(Enum):
    NORMAL = 0
    RECREATE = 1

  def __init__(self, environment, **kwargs) -> None:
    assert not environment is None, "environment must be a valid Environment object. Simulator constructor failed"
    assert "drones" in kwargs or "trajectories" in kwargs,\
      "Either drones = list of {'id': drone_id, 'state': state} and controller = Controller()\n\
      or trajectories = dict of {time_step: list[{drone_id, drone_state}]} must be supplied as kwargs.\n\
      Simulator constructor failed"

    self.log_sim = kwargs.get("log_sim", False)

    self.environment = environment

    if "drones" in kwargs:
      assert "controller" in kwargs,\
        "controller must be supplied as a keyword argument when drone trajectories are not known. Simulator constructor failed"

      self.state = self.SimulatorState.NORMAL


      self.drones = Simulator.__load_drones(kwargs["drones"])
      self.drone_sensor_data = {}
      self.drone_states = {}
      self.commands = {d.id: np.zeros((6, 1)) for d in self.drones}

      self.controller = kwargs.get("controller", None)

      self.drone_swarm = DroneSwarm(self.drones,self.controller)

      self.com_channel = CommunicationChannel(
        lambda sender, recipient: kwargs.get("com_filter", lambda s, r: True)(sender, recipient),
        delay = kwargs.get("com_delay", None),
        packet_loss = kwargs.get("com_packet_loss", None)
      )

      if self.log_sim:
        self.logger = Logger(drones=self.drones, environment=self.environment)
  
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
  
  def sim_step(self, step_length, time):
    """
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
    """

    self.drone_swarm.sim_step(step_length, self.environment)
    for d in self.drone_swarm.drones:
      self.logger.write_to_log(d.get_time_entry(), time)