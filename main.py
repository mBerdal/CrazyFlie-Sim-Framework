from environment import Environment
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


#mplstyle.use('fast')
if __name__ == "__main__":
  x_max, y_max = (200, 200)
  e = Environment([[x == 0 or x == x_max-1 or y == 0 or y == y_max-1 for x in range(x_max)] for y in range(y_max)])

  cfs = [None]*4
  for i in range(len(cfs)):
    cfs[i] = CrazyFlie(i,states[i])
    cfs[i].update_command(commands[i])

  c = CommunicationChannel()
  c.send_msg(cfs[0], cfs[1:4], "JADDA")

  run_animation(e, cfs)