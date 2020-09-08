from environment import Environment
from range_sensor import RangeSensor
from crazy_flie import CrazyFlie
from communication import CommunicationChannel, CommunicationNode

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def run_animation(environment, crazy_flies):
  fig, ax = plt.subplots()
  ax.axis("equal")
  environment.plot(ax)
  for cf in crazy_flies:
    cf.plot(ax,environment)
  def animate(i):
    for cf in crazy_flies:
      cf.do_step(e, 0.10)
      cf.update_plot(environment)
  an = animation.FuncAnimation(fig, animate, frames=10, repeat=False)
  plt.show()

max_range = 4
range_res = 0.001
pos_body = np.array([0,0,0]).reshape(3,1)
arc_angle = np.deg2rad(27)
num_beams = 9
arc_res = np.deg2rad(27/num_beams)

attitude_body1 = np.array([0, 0, np.pi]).reshape(3, 1)
sensor1 = RangeSensor(max_range,range_res,arc_angle,arc_res,pos_body,attitude_body1)
attitude_body2 = np.array([0,0,np.pi/2]).reshape(3,1)
sensor2 = RangeSensor(max_range, range_res, arc_angle, arc_res, pos_body, attitude_body2)
attitude_body3 = np.array([0, 0, -np.pi / 2]).reshape(3, 1)
sensor3 = RangeSensor(max_range, range_res, arc_angle, arc_res, pos_body, attitude_body3)
attitude_body4 = np.array([0, 0, 0]).reshape(3, 1)
sensor4 = RangeSensor(max_range, range_res, arc_angle, arc_res, pos_body, attitude_body4)
sensors = [sensor1,sensor2,sensor3,sensor4]


state1 = np.array([0.5,0.5,0,0,0,0]).reshape((6,1))
state2 = np.array([15.5,0.5,0,0,0,0]).reshape((6,1))
state3 = np.array([0.5,15.5,0,0,0,0]).reshape((6,1))
state4 = np.array([15.5,15.5,0,0,0,0]).reshape((6,1))
states = [state1,state2,state3,state4]
if __name__ == "__main__":
  x_max, y_max = (200, 200)
  e = Environment([[x == 0 or x == x_max-1 or y == 0 or y == y_max-1 for x in range(x_max)] for y in range(y_max)])

  cfs = [None]*4
  for i in range(len(cfs)):
    cfs[i] = CrazyFlie(states[i], sensors)

  c = CommunicationChannel()
  c.distribute_msg(cfs[0], cfs[1:3], "JADDA")

  run_animation(e, cfs)