from environment.environment import Environment
from drone_swarm.drone.crazy_flie import CrazyFlie
from drone_swarm.drone_swarm import DroneSwarm
from communication import CommunicationNode, CommunicationChannel
from logger.logger import Logger
from utils.raytracing import multi_ray_intersect_triangle

from sensor.range_sensor import RangeSensor

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

class Simulator(CommunicationNode):

  def __init__(self, **kwargs) -> None:
    self.log_sim = kwargs.get("log_sim", True)
    self.environment = kwargs.get("environment", None)
    self.drones = kwargs.get("drones", None)
    self.logger = kwargs.get("logger", None)
    self.controller = kwargs.get("controller", None)

    self.drone_swarm = DroneSwarm(self.drones, self.controller)

    self.com_channel = CommunicationChannel(
      lambda sender, recipient: kwargs.get("com_filter", lambda s, r: True)(sender, recipient),
      delay = kwargs.get("com_delay", None),
      packet_loss = kwargs.get("com_packet_loss", None)
    )

    if self.logger is None and self.log_sim:
      self.logger = Logger(drones=self.drones, environment=self.environment)
  
  def recv_msg(self, msg):
    if not msg is None:
      msg()

  def simulate(self, step_length_seconds, end_time_seconds):
    self.time_line = np.arange(0, end_time_seconds, step_length_seconds)
    self.step_length = step_length_seconds
    for time in self.time_line:
      self.__sim_step(step_length_seconds, time)
  
  def __sim_step(self, step_length, time):
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

  def visualize(self):
    fig, axis = plt.subplots(1)
    figs = {
      drone_id: [RangeSensor.init_plot(
          axis,
          self.logger.get_drone_state_at_time(drone_id, 0),
          self.logger.get_drone_sensor_specs(drone_id, sensor_idx).sensor_pos_bdy,
          self.logger.get_drone_sensor_specs(drone_id, sensor_idx).sensor_attitude_bdy,
          self.logger.get_drone_sensor_specs(drone_id, sensor_idx).max_range,
          self.logger.get_drone_sensor_specs(drone_id, sensor_idx).arc_angle,
          self.logger.get_drone_sensor_measurements_at_time(drone_id, sensor_idx, 0)
        ) for sensor_idx in range(self.logger.get_num_drone_sensors(drone_id))
      ] for drone_id in self.logger.get_drone_ids()
    }

    def animate(t):
      updated_figs = []
      for drone_id in self.logger.get_drone_ids():
        for index, fig in enumerate(figs[drone_id]):
          updated_figs.append(
              RangeSensor.update_plot(
                fig,
                self.logger.get_drone_state_at_time(drone_id, t),
                self.logger.get_drone_sensor_specs(drone_id, index).sensor_pos_bdy,
                self.logger.get_drone_sensor_specs(drone_id, index).sensor_attitude_bdy,
                self.logger.get_drone_sensor_specs(drone_id, index).max_range,
                self.logger.get_drone_sensor_specs(drone_id, index).arc_angle,
                self.logger.get_drone_sensor_measurements_at_time(drone_id, index, t)
            )
          )
      return updated_figs

    axis.axis("equal")

    anim = FuncAnimation(fig, animate, frames=self.time_line, interval=self.step_length*1000, repeat=False, blit=True)
    plt.show()