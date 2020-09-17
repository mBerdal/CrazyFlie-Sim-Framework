from environment import Environment
from utils.json_utils import read_json, write_json

from typing import Dict, List, Tuple
import numpy as np

class Logger():
  """
  Class for storing/fetching information created during simulations.
  ...

  Attributes
  ----------
  environment: Environment/String/None

    If None the Logger is in read_only mode. If not it points to either a file
    containing a json dictionary representing an Environment object or an Environment
    object itself.

  trajectories: dict{some_float: list[dict{id: str, state: np.ndarray}]}

    Dictionary where keys are timesteps [sec] and values are lists containing
    dictionaries with an ID for a CrazyFlie drone and the state of the given
    drone at said timestep

  Methods
  ----------
  __init__(environment): None

    Takes as optional argument an environment. If environment is None the logger is in
    read-only mode, if not the logger is in write only mode. environment can either be
    an Environment object or the location of a file containing a json representation of
    an Environment object.
  
  read_log(filename): Environment, Dict

    **only available on read-only mode**
    Takes as argument a filename, and retrieves an enviroment object specified in the file and
    a dictionary containing timesteps as keys and IDs and states of drones at given timestep as values.

  write_to_log(timestep, drone_id, drone_state, measurements): None

    **only available in write-only mode**
    Takes as argument a timestep [sec], an ID for a drone, the state (n-by-1 matrix) of the drone, and the
    sensor measurements at the timestep specified and stores it.

  save_log(filename): None

    **only available in read-only mode**
    Takes as argument a filename and writes a json dict containing an entry for the environment variable of
    the Logger and an entry for the trajectories of the Logger.

"""

  def __init__(self, drones = [], environment = None):
    self.environment = environment
    self.trajectories = {}
    self.drones_info = {}
    for drone in drones:
      self.drones_info = {**self.drones_info, **drone.get_specs_dict()}
      
  def read_log(self, filename: str = "") -> Tuple[Environment, Dict[float, List[Dict[str, np.ndarray]]]]:
    assert self.environment is None, "Logger instantiated with environment is in write-only mode. read_log failed"
    try:
      data = read_json(filename)

      drones = data["drones"]

      trajectories = data["trajectories"]
      for timestep in trajectories:
        for entry in trajectories[timestep]:
          entry["state"] = np.array(entry["state"])

      env_dict = data["environment"]
      env_objects = None
      if "path" in env_dict.keys():
        env_objects = read_json(env_dict["path"])
      elif "objects" in env_dict.keys():
        env_objects = [{"shape": obj["shape"], "points": np.array(obj["points"])} for obj in env_dict["objects"]]
      else:
        raise Warning("Invalid environment argument in file")
      
      return drones, Environment(env_objects), trajectories
      
    except FileNotFoundError as fnfe:
      print(fnfe)
      return None, None
  
  def __is_valid_drone_id(self, drone_id):
    for d_info in self.drones_info:
      if d_info["id"] == drone_id: return True
    return False
    
  def write_to_log(self, timestep: float, drone_id: str, drone_state: np.ndarray, measurements) -> None:
    assert not self.environment is None, "Logger does not have a set environment and is therefore in read-only mode. write_to_log failed"
    assert self.__is_valid_drone_id, f"agent with id {drone_id} not registered in log"

    id_state_meas_dict = {"id": drone_id, }
    try:
      self.trajectories[timestep][drone_id] = {"state": drone_state.tolist(), "measurements": measurements}
    except KeyError:
      self.trajectories[timestep] = {drone_id: {"state": drone_state.tolist(), "measurements": measurements}}

  def save_log(self, filename):
    assert not self.environment is None, "Logger does not have a set environment can therefore not be saved. save_log failed"
    assert not filename is None, "filename not supplied. save_log failed"
    assert isinstance(self.environment, str)\
      or isinstance(self.environment, Environment), "environment must be either a string or an Environment object. save_log failed"
    write_json(filename, {
      "drones": self.drones_info,
      "environment": {"path": self.environment} if isinstance(self.environment, str) else {
        "objects": [
          {
            "shape": obj["shape"],
            "points": obj["points"].tolist()
          } for obj in self.environment.objects
        ]
      },
      "trajectories": self.trajectories
    })