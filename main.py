from environment import Environment
from range_sensor import RangeSensor
from crazy_flie import CrazyFlie

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

if __name__ == "__main__":
  e = Environment([[x == 0 or x == 200-1 or y == 0 or y == 200-1 for x in range(200)] for y in range(200)])
  
  range_sensors = [
    RangeSensor(4, 0.079, np.deg2rad(27), np.deg2rad(0.1), np.array([0.08 if i == 0 or i == 3 else -0.08, 0.08 if i<2 else -0.08]), np.deg2rad(45 + 90*i))
  for i in range(4)]
  c = CrazyFlie(np.array([0.0, 0.0]), range_sensors)
  
  fig, ax = plt.subplots()
  e.plot(ax)
  c.plot(ax)

  def animate(i):
    c.do_step(e, 0.10)
    c.update_plot()
    return [c.fig] + [s.fig for s in c.sensors]
  
  ani = animation.FuncAnimation(fig, animate, frames=10, blit=True, repeat=False)
  plt.show()