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
    for ind, cf in enumerate(crazy_flies):
      cf.update_state(0.10)
      readings = cf.read_sensors(environment,vector_format=False)
      #print("Sensor Readings CF:",cf.id)
      #print(readings)
      cf.update_plot(environment)
  an = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
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


state1 = np.array([5,5,0,0,0,0]).reshape((6,1))
state2 = np.array([10,5,0,0,0,0]).reshape((6,1))
state3 = np.array([5,10,0,0,0,0]).reshape((6,1))
state4 = np.array([10,10,0,0,0,0]).reshape((6,1))
states = [state1,state2,state3,state4]

command1 = np.array([0,0,0,0,0,np.deg2rad(10)]).reshape(6,1)
command2 = np.array([1,0,0,0,0,np.deg2rad(10)]).reshape(6,1)
command3 = np.array([-1,0,0,0,0,np.deg2rad(10)]).reshape(6,1)
command4 = np.array([0.5,0,0,0,0,np.deg2rad(10)]).reshape(6,1)
commands = [command1,command2,command3,command4]

states = [state1,state2,state3,state4]

state_dot = np.zeros((6,1))
acc_limits_lower = -np.array([1,1,1,np.deg2rad(10),np.deg2rad(10),np.deg2rad(10)]).reshape(6,1)
acc_limits_upper = np.array([1,1,1,np.deg2rad(10),np.deg2rad(10),np.deg2rad(10)]).reshape(6,1)

if __name__ == "__main__":
  x_max, y_max = (200, 200)
  e = Environment([[x == 0 or x == x_max-1 or y == 0 or y == y_max-1 for x in range(x_max)] for y in range(y_max)])

  cfs = [None]*4
  for i in range(len(cfs)):
    sensors = [RangeSensor(max_range,range_res,arc_angle,arc_res,pos_body,attitude_body1),
               RangeSensor(max_range, range_res, arc_angle, arc_res, pos_body, attitude_body2),
               RangeSensor(max_range, range_res, arc_angle, arc_res, pos_body, attitude_body3),
               RangeSensor(max_range, range_res, arc_angle, arc_res, pos_body, attitude_body4)]
    cfs[i] = CrazyFlie(i,states[i],state_dot,sensors,acc_limits_lower,acc_limits_upper)
    cfs[i].update_command(commands[i])

  c = CommunicationChannel()
  c.send_msg(cfs[0], cfs[1:4], "JADDA")

  run_animation(e, cfs)