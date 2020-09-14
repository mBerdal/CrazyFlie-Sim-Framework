from environment import Environment
from crazy_flie import CrazyFlie
from communication import CommunicationChannel, CommunicationNode
from simulator import Simulator
from controller import SwarmController

import matplotlib.pyplot as plt
import numpy as np


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


points1 = np.array([[0,0,16,16],[0,0,0,0],[0,3,0,3]])
points2 = np.array([[0,0,0,0],[0,0,16,16],[0,3,0,3]])
points3 = np.array([[16,16,16,16],[0,0,16,16],[0,3,0,3]])
points4 = np.array([[0,0,16,16],[16,16,16,16],[0,3,0,3]])
points5 = np.array([[0,0,10,10],[10,10,10,10],[0,3,0,3]])

obj1 = {"shape": "rectangle","points":points1}
obj2 = {"shape": "rectangle", "points": points2}
obj3 = {"shape": "rectangle", "points": points3}
obj4 = {"shape": "rectangle", "points": points4}
obj5 = {"shape": "rectangle", "points": points5}

objects = [obj1,obj2,obj3,obj4]

num_objects = 10
new_objects = num_objects-len(objects)

z1 = 0
z2 = 3
for _ in range(new_objects):
  x1 = np.random.randint(0,16)
  x2 = np.random.randint(0,16)
  y1 = np.random.randint(0,16)
  y2 = np.random.randint(0,16)

  points = np.array([[x1,x1,x2,x2],[y1,y1,y2,y2],[z1,z2,z1,z2]])
  obj_tmp = {"shape": "rectangle", "points": points}
  objects.append(obj_tmp)

env = Environment(objects)

plot = True
if __name__ == "__main__":
  c = SwarmController(drones,set_points)
  s = Simulator(env, drones=drones, controller=c)

  if plot:
    fig, ax = plt.subplots()
    ax.axis("equal")
    env.plot(ax)
    for cf in s.drones:
      cf.plot(ax, env,plot_sensors=True)
    plt.show(block=False)

  for i in range(300):
    if i == 70:
      c.update_set_points(new_setpoints)
    s.sim_step(0.05)
    if plot:
      for cf in s.drones:
        cf.update_plot(env,plot_sensors=True)
      fig.canvas.draw()
      fig.canvas.flush_events()
