from environment.environment import Environment
from environment.obstacle import Obstacle
from drone_swarm.drone.crazy_flie import CrazyFlie
from communication import CommunicationChannel, CommunicationNode
from simulator import Simulator
from controller import SwarmController
from sensor.range_sensor import RangeSensor
from logger.logger import Logger
import numpy as np

np.random.seed(0)
num_drones = 2
plot = False
plot_rays = True
steps = 300

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
  drones.append(CrazyFlie(i, state, state_noise_generator=lambda: np.random.uniform(-0.1, 0.1, state.shape)))
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

points_array = [
  np.array([[0,0,16,16],[0,0,0,0],[0,3,0,3]]),
  np.array([[0,0,0,0],[0,0,16,16],[0,3,0,3]]),
  np.array([[16,16,16,16],[0,0,16,16],[0,3,0,3]]),
  np.array([[0,0,16,16],[16,16,16,16],[0,3,0,3]]),
  np.array([[0,0,10,10],[10,10,10,10],[0,3,0,3]])
]

objects = [Obstacle("rectangle", points) for points in points_array]

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
  objects.append(Obstacle("rectangle", points))

env = Environment()

for o in objects:
  env.add_obstacle(o)


if __name__ == "__main__":
  c = SwarmController(drones,set_points)
  s = Simulator(environment=env, drones=drones, controller=c, com_delay=0.1, log_to_file="test_0.json", env_to_file="env_test_0.json")
  s.simulate(0.05, 10)

  k = Logger()
  k.load_from_file("test_0.json")
  s = Simulator(logger = k)
  s.visualize()

  