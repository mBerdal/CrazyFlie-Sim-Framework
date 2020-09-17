from environment import Environment
from crazy_flie import CrazyFlie
from range_sensor import RangeSensor
from communication import CommunicationNode, CommunicationChannel
from logger import Logger
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

      self.com_channel = CommunicationChannel(
        lambda sender, recipient: kwargs["com_filter"](sender, recipient) if "com_filter" in kwargs else True,
        delay = kwargs["com_delay"] if "com_delay" in kwargs else None,
        packet_loss = kwargs["com_packet_loss"] if "com_packet_loss" in kwargs else None
      )

      if self.log_sim:
        self.logger = Logger(drones=self.drones, environment=self.environment)
  
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
    self.get_drone_sensors()
    self.get_drone_states()
    self.commands = self.controller.get_commands(self.drone_states, self.drone_sensor_data)
    for d in self.drones:
      d.update_command(self.commands[d.id])
      state = d.update_state(step_length)
      self.logger.write_to_log(time, d.id, state, None)

  def get_drone_sensors(self):
    readings, orgins, idx_drones = self.read_range_sensors()
    self.drone_sensor_data = {}
    for d in self.drones:
      self.drone_sensor_data[d.id] = readings[:,idx_drones[d.id]["start"]:idx_drones[d.id]["end"]]

  def get_drone_states(self):
    self.drone_states = {}
    for d in self.drones:
      self.drone_states[d.id] = d.state

  def read_range_sensors(self):
    rays, orgins, idx_drones = self.get_rays()
    rays = rays.transpose()
    orgins = orgins.transpose()
    t_min = np.ones(rays.shape[0])*np.inf
    for obj in self.environment.get_objects():
      t = multi_ray_intersect_triangle(orgins,rays,obj["points"],4)
      t_min = np.minimum(t_min,t)
    t_min[t_min==np.inf] = 4
    reading = t_min.reshape(-1,1)*rays
    return reading, orgins , idx_drones

  def plot(self,ax,plot_sensors=False):
    self.figs_drones = []

    self.environment.plot(ax)
    for d in self.drones:
      self.figs_drones.append(
      ax.plot(d.state[0],d.state[1],"go")
      )
    if plot_sensors:
      self.figs_rays = []
      rays, orgins, idx_drones = self.read_range_sensors()
      for i in range(rays.shape[0]):
        ray = rays[i,:]
        orgin = orgins[i,:]
        self.figs_rays.append(
        ax.plot([orgin[0],orgin[0]+ray[0]],[orgin[1],orgin[1]+ray[1]],"r")
        )

  def update_plot(self,ax,plot_sensors=False):
    for ind, d in enumerate(self.drones):
      self.figs_drones[ind][0].set_data(d.state[0],d.state[1])
    if plot_sensors:
      rays, orgins, idx_drones = self.read_range_sensors()
      for i in range(rays.shape[0]):
        ray = rays[i, :]
        orgin = orgins[i, :]
        self.figs_rays[i][0].set_data(
          [orgin[0], orgin[0] + ray[0]], [orgin[1], orgin[1] + ray[1]]
        )

  def get_rays(self):
    rays = []
    orgins = []
    idx_drones = {}
    idx = 0
    for d in self.drones:
      r, o = d.get_sensor_rays()
      rays.append(r)
      orgins.append(o)
      idx_drones[d.id] = {"start": idx, "end": r.shape[1]}
      idx = idx + r.shape[1]
    rays = np.concatenate(rays,axis=1)
    orgins = np.concatenate(orgins,axis=1)
    return rays, orgins, idx_drones
