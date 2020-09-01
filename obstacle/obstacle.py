
class Obstacle():
  def __init__(self, fun) -> None:
    self.fun = fun

  def is_at_point(self, x, y) -> bool:
    return self.fun(x, y)