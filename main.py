from environment import Environment
from crazy_flie import CrazyFlie
from communication import CommunicationChannel, CommunicationNode
from simulator import Simulator
from controller import SwarmController

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
      cf.update_plot(environment)
  an = animation.FuncAnimation(fig, animate, frames=100, repeat=False)
  plt.show()

state1 = np.array([5,5,0,0,0,0]).reshape((6,1))
state2 = np.array([10,5,0,0,0,0]).reshape((6,1))
state3 = np.array([2,5,0,0,0,0]).reshape((6,1))
state4 = np.array([10,10,0,0,0,0]).reshape((6,1))
states = [state1,state2,state3,state4]

drones = [{"id": 1, "state": state1},
          {"id": 2, "state": state2}]
set_points = [state2[0:3,:],state4[0:3,:]]
new_setpoints = [{"id":2, "set_point": state1[0:3,:]},
                 {"id":1, "set_point": state3[0:3,:]}]

if __name__ == "__main__":
  x_max, y_max = (200, 200)
  e = Environment([[x == 0 or x == x_max-1 or y == 0 or y == y_max-1 for x in range(x_max)] for y in range(y_max)])
  c = SwarmController(drones,set_points)
  s = Simulator(e, drones=drones, controller=c)

  fig, ax = plt.subplots()
  ax.axis("equal")
  e.plot(ax)
  for cf in s.drones:
    cf.plot(ax, e,plot_sensors=True)
  plt.show(block=False)
  for i in range(300):
    if i == 70:
      c.update_set_points(new_setpoints)
    s.sim_step(0.05)

    for cf in s.drones:
      cf.update_plot(e,plot_sensors=True)
    fig.canvas.draw()
    fig.canvas.flush_events()