from environment import Environment
from range_sensor import RangeSensor
from crazy_flie import CrazyFlie

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
  x_max, y_max = (200, 200)
  e = Environment([[x == 0 or x == x_max-1 or y == 0 or y == y_max-1 for x in range(x_max)] for y in range(y_max)])

  cfs = [None]*4
  for i in range(len(cfs)):
    range_sensors = [
      RangeSensor(4, 0.079, np.deg2rad(27), np.deg2rad(27/3), np.array([0.08 if i == 0 or i == 3 else -0.08, 0.08 if i<2 else -0.08]), np.deg2rad(45 + 90*i))
    for i in range(4)]
    cfs[i] = CrazyFlie(np.array([0.08+0.16 if i < 2 else 15.92-0.16, 0.08+0.16 if i == 0 or i == 2 else 15.92-0.16]), range_sensors)
  
  fig, ax = plt.subplots()
  ax.axis("equal")
  e.plot(ax)
  for cf in cfs: cf.plot(ax)
  def animate(i):
    for cf in cfs:
      cf.do_step(e, 0.10)
      cf.update_plot()


  ani = animation.FuncAnimation(fig, animate, frames=10, repeat=False)

  plt.show()