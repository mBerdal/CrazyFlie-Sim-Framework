from environment import Environment
from crazy_flie import CrazyFlie
from communication import CommunicationNode, CommunicationChannel
from logger import Logger
from drone_swarm import DroneSwarm
from utils.raytracing import multi_ray_intersect_triangle
import matplotlib.pyplot as plt

import numpy as np
from enum import Enum

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

      self.controller = kwargs["controller"]

      self.drone_swarm = DroneSwarm(self.drones,self.controller)

      self.com_channel = CommunicationChannel(
        lambda sender, recipient: kwargs["com_filter"](sender, recipient) if "com_filter" in kwargs else True,
        delay = kwargs["com_delay"] if "com_delay" in kwargs else None,
        packet_loss = kwargs["com_packet_loss"] if "com_packet_loss" in kwargs else None
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

    drone_states, sensor_readings = self.drone_swarm.sim_step(step_length, self.environment)
    for d_id, d_state in drone_states.items():
      self.logger.write_to_log(time, d_id, d_state, sensor_readings[d_id])