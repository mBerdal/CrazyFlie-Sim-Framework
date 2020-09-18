from environment import Environment
from log_entry import EntryType
from utils.json_utils import read_json, write_json

from typing import Dict, List, Tuple
import numpy as np

from copy import deepcopy
import importlib

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
    self.log = {"environment": environment, "drones": {}}
    for d in drones:
      d_info_entry = d.get_info_entry()
      self.log["drones"][d_info_entry.id] = {"info": d_info_entry}
      
  def read_log(self, filename: str = ""):
    try:
      data = read_json(filename)

      drones_dict = data["drones"]

      drones = []
      for id, drone in drones_dict.items():
        d_class = getattr(importlib.import_module(drone["info"]["module"]), drone["info"]["cls"])
        d_args = dict(filter(lambda elem: elem[0] != "module" and elem[0] != "cls", drone["info"].items()))

        sensors = []
        for s in drone["info"]["sensors"]:
          s_class = getattr(importlib.import_module(s["module"]), s["cls"])
          s_args = dict(filter(lambda elem: elem[0] != "module" and elem[0] != "cls", s.items()))
          s_args = {k: np.array(v) if isinstance(v, list) else v for k, v in s_args.items()}
          sensors.append(s_class(**s_args))
          
        drones.append(d_class(id = id, state = np.array(drone["trajectory"]["0"]["state"]), **d_args))
      print(drones)
          


      env_dict = data["environment"]
      env_objects = None
      if "path" in env_dict.keys():
        env_objects = read_json(env_dict["path"])
      elif "objects" in env_dict.keys():
        env_objects = [{"shape": obj["shape"], "points": np.array(obj["points"])} for obj in env_dict["objects"]]
      else:
        raise Warning("Invalid environment argument in file")
      
      return drones, Environment(env_objects)
      
    except FileNotFoundError as fnfe:
      print(fnfe)
      return None, None
    
  def write_to_log(self, log_entry, timestep: float) -> None:
    if log_entry.type == EntryType.INFO:
      self.log["drones"][log_entry.id]["info"] = log_entry
    elif log_entry.type == EntryType.TIME:
      try:
        self.log["drones"][log_entry.id]["trajectory"][timestep] = log_entry
      except KeyError:
        self.log["drones"][log_entry.id]["trajectory"] = {timestep: log_entry}

  def save_log(self, filename):
    assert not filename is None, "filename not supplied. save_log failed"

    lg = deepcopy(self.log)
    for _, drone_log in lg["drones"].items():
      drone_log["info"] = drone_log["info"].to_JSONable()
      for step, step_log in drone_log["trajectory"].items():
        drone_log["trajectory"][step] = step_log.to_JSONable()

    if isinstance(self.log["environment"], str):
      lg["environment"] = {"path": self.log["environment"]}
    else:
      lg["environment"] = self.log["environment"].to_JSONable()
    write_json(filename, lg)