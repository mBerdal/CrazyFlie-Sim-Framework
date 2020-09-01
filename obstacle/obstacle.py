
class Obstacle():

  """
  Takes as parameter a function 'fun' which takes as parameters coordinates (x, y),
  and returns a boolean which indicates wether or not the obstacle is at (x, y)
  """
  def __init__(self, fun) -> None:
    self.fun = fun

  def is_at_point(self, x, y) -> bool:
    return self.fun(x, y)