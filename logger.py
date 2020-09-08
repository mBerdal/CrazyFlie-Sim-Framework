from environment import Environment
from utils.json_utils import read_json

from typing import Dict, List, Tuple
import numpy as np

class Logger():
  def __init__(self, environment: Environment = None):
    self.environment = environment

  def read_log(self, filename: str = "") -> Tuple[Environment, Dict[float, List[Dict[str, np.ndarray]]]]:
    assert self.environment is None, "Logger instantiated with environment is in write-only mode. read_log failed"
    try:
      data = read_json(filename)
      trajectories = data["trajectories"]
      environment = data["environment"]
      if "filename" in environment.keys():
        environment = read_json(environment["filename"])

      return Environment(environment["grid"], (environment["res_x"], environment["res_y"])), trajectories
      
    except FileNotFoundError as f:
      print(f)
      return None, None