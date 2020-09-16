from environment import Environment
from crazy_flie import CrazyFlie
from communication import CommunicationChannel, CommunicationNode
from simulator import Simulator
from controller import SwarmController

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)
num_drones = 1000
plot = False
plot_rays = True
steps = 1000

x_lim_l = 0
x_lim_u = 16

y_lim_l = 0
y_lim_u = 16

z_lim_l = 0
z_lim_u = 3

yaw_lim_l = -np.pi/4
yaw_lim_u = np.pi/4

drones = []
set_points = []

for i in range(num_drones):
  x = np.random.uniform(x_lim_l,x_lim_u)
  y = np.random.uniform(y_lim_l,y_lim_u)
  z = np.random.uniform(z_lim_l,z_lim_u)
  yaw = np.random.uniform(yaw_lim_l,yaw_lim_u)
  state = np.array([x,y,z,0,0,yaw]).reshape((6,1))
  drones.append({"id":i,"state":state})
  x_set = np.random.uniform(x_lim_l,x_lim_u)
  y_set = np.random.uniform(y_lim_l,y_lim_u)
  z_set = np.random.uniform(z_lim_l,z_lim_u)
  sp = np.array([x_set,y_set,z_set]).reshape(3,1)
  set_points.append(sp)

def generate_set_points():
  new_set_points = []
  for i in range(num_drones):
    x_set = np.random.uniform(x_lim_l, x_lim_u)
    y_set = np.random.uniform(y_lim_l, y_lim_u)
    z_set = np.random.uniform(z_lim_l, z_lim_u)
    sp = np.array([x_set, y_set, z_set]).reshape(3, 1)
    new_set_points.append({"id": i,"set_point":sp})
  return new_set_points

"""
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
"""

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

num_objects = 4
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

env = Environment()

for o in objects:
  env.add_object(o)


if __name__ == "__main__":
  c = SwarmController(drones,set_points)
  s = Simulator(env, drones=drones, controller=c)


  if plot:
    fig, ax = plt.subplots()
    ax.axis("equal")
    s.plot(ax,plot_sensors=plot_rays)
    plt.show(block=False)

  print("Starting simulation!")
  for i in range(steps):

    if i % 100 == 0 and i != 0:
      new_sp = generate_set_points()
      c.update_set_points(new_sp)
    print("Simulating step:",i+1,"of",steps)
    s.sim_step(0.05)
    if plot:
      s.update_plot(ax,plot_sensors=plot_rays)
      fig.canvas.draw()
      fig.canvas.flush_events()