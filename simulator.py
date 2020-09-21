from environment.environment import Environment
from drone_swarm.drone.crazy_flie import CrazyFlie
from drone_swarm.drone_swarm import DroneSwarm
from communication import CommunicationChannel
from logger.logger import Logger
from utils.raytracing import multi_ray_intersect_triangle

from plottable import Plottable

from sensor.range_sensor import RangeSensor

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial

class Simulator():

  def __init__(self, **kwargs) -> None:
    self.to_file = kwargs.get("to_file", None)
    self.environment = kwargs.get("environment", None)
    self.drones = kwargs.get("drones", None)
    self.logger = kwargs.get("logger", None)
    self.controller = kwargs.get("controller", None)

    assert (not self.logger is None) or not (self.drones is None or self.controller is None or self.environment is None)

    if not self.logger is None:
      self.environment = self.logger.get_environment()
      self.step_length = self.logger.get_log_time_step()
      self.time_line = np.arange(0, self.logger.get_log_end_time() + self.step_length, self.step_length)
    else:
      com_channel = CommunicationChannel(
        lambda sender, recipient: kwargs.get("com_filter", lambda s, r: True)(sender, recipient),
        delay = kwargs.get("com_delay", 0),
        packet_loss = kwargs.get("com_packet_loss", 0)
      )
      self.drone_swarm = DroneSwarm(self.drones, self.controller, com_channel)
      self.logger = Logger(drones=self.drones, environment=self.environment)


  def simulate(self, step_length_seconds, end_time_seconds):
    self.time_line = np.arange(0, end_time_seconds + step_length_seconds, step_length_seconds)
    self.step_length = step_length_seconds
    for time in self.time_line:
      self.__sim_step(step_length_seconds, time)

    if not self.to_file is None:
      self.logger.save_log(self.to_file)
  
  def __sim_step(self, step_length, time):
    self.drone_swarm.sim_step(step_length, self.environment)
    coords = None
    if not time % 3:
      coords = np.empty((3, len(self.drone_swarm.drones))).reshape(len(self.drone_swarm.drones),3) #3D
    k = 0
    for d in self.drone_swarm.drones:
      self.logger.log_time_step(d.get_time_entry(), time)
      if coords is not None:
        coords[k,:] = d.state[0:3].reshape(1,3)
        k = k+1
    if coords is not None:
      self.collision_check(coords)
  
  def collision_check(self, coords: dict):
    dist_mat = scipy.spatial.distance.pdist(coords)
    square_mat = scipy.spatial.distance.squareform(dist_mat)
    collision_indices = np.argwhere(square_mat < 2)
    
    for indices in collision_indices:
      if indices[0] != indices[1]:
        print(f"WARNING: drone{indices[0]} and drone{indices[1]} collided")

  def visualize(self):
    fig, axis = plt.subplots(1)
    figs = {}
    for drone_id in self.logger.get_drone_ids():
      figs[drone_id] = {
        "drone": self.logger.get_drone_class(drone_id).init_plot(axis, **self.logger.get_drone_plotting_kwargs(drone_id, self.time_line.item(0))),
        "sensors": {}
      }
      for sensor_idx in range(self.logger.get_num_drone_sensors(drone_id)):
        sensor_cls = self.logger.get_sensor_class(drone_id, sensor_idx)
        if issubclass(sensor_cls, Plottable):
          figs[drone_id]["sensors"][sensor_idx] = sensor_cls.init_plot(
            axis,
            **self.logger.get_sensor_plotting_kwargs(drone_id, sensor_idx, self.time_line.item(0))
          )

    self.environment.plot(axis)

    def animate(t):
      axis.set_title(f't = {t:.2f}')
      axis.axis("equal")
      for drone_id in self.logger.get_drone_ids():
        self.logger.get_drone_class(drone_id).update_plot(
          figs[drone_id]["drone"],
          **self.logger.get_drone_plotting_kwargs(drone_id, t)
        )
        for sensor_idx, fig in figs[drone_id]["sensors"].items():
          sensor_cls = self.logger.get_sensor_class(drone_id, sensor_idx)
          sensor_cls.update_plot(
          fig,
          **self.logger.get_sensor_plotting_kwargs(drone_id, sensor_idx, t)
          )


    anim = FuncAnimation(fig, animate, frames=self.time_line, interval=self.step_length*1000, repeat=False, blit=False)
    plt.draw()
    plt.show()