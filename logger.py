from environment import Environment
from utils.json_utils import read_json

from typing import Dict, List, Tuple
import numpy as np

class Logger():
  def __init__(self, environment):
    self.environment = environment
    self.trajectories = {}

  def read_log(self, filename: str = "") -> Tuple[Environment, Dict[float, List[Dict[str, np.ndarray]]]]:
    assert self.environment is None, "Logger instantiated with environment is in write-only mode. read_log failed"
    try:
      data = read_json(filename)
      trajectories = data["trajectories"]
      environment = data["environment"]
      if "filename" in environment.keys():
        environment = read_json(environment["filename"])

      return Environment(environment["grid"], (environment["res_x"], environment["res_y"])), trajectories
      
    except FileNotFoundError as fnfe:
      print(fnfe)
      return None, None
    
  def write_to_log(self, timestep: float, cfID: int, cfState: np.ndarray) -> None:
    assert not self.environment is None, "Logger does not have a set environment and is therefore in read-only mode. write_to_log failed"
    id_state_dict = {"id": cfID, "state": cfState}
    try:
      self.trajectories[timestep].append(id_state_dict)
    except KeyError:
      self.trajectories[timestep] = [id_state_dict]
