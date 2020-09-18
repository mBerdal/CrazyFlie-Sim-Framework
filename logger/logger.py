from environment.environment import Environment
from logger.log_entry import EntryType
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
    self.log = {"environment": None, "drones": {}}
    self.__log_environment(environment)
    self.__log_info(drones)

  def __log_environment(self, environment):
    self.log["environment"] = environment

  def __log_info(self, drones):
    for d in drones:
      self.log["drones"][d.id] = {"info": d.get_info_entry()}
      
  def load_from_file(self, filename: str = ""):
    try:
      data = read_json(filename)

      drones_dict = data.get("drones", None)
      assert not drones_dict is None, f"Drones not specified in file {filename}. Failed to read."

      env_dict = data.get("environment", None)
      assert not env_dict is None, f"Environment not specified in file {filename}. Failed to read."

      assert "trajectory" in dict(next(iter(drones_dict.values()))).keys(), f"Trajectories not specified in file {filename}. Failed to read."
      traj_dict = {id: drone_dict["trajectory"] for id, drone_dict in drones_dict.items()}

      if "path" in env_dict.keys():
        env_dict = read_json(env_dict["path"])
      self.__log_environment(Environment([{"shape": obj["shape"], "points": np.array(obj["points"])} for obj in env_dict["objects"]]))

      drones = []
      for id, drone in drones_dict.items():
        d_class = getattr(importlib.import_module(f"drone_swarm.drone.{drone['info']['module']}"), drone["info"]["cls"])
        d_args = dict(filter(lambda elem: elem[0] != "module" and elem[0] != "cls", drone["info"].items()))

        sensors = []
        for s in drone["info"]["sensors"]:
          s_class = getattr(importlib.import_module(f"sensor.{s['module']}"), s['cls'])
          s_args = {
            k: np.array(v) if isinstance(v, list) else v
            for k, v in dict(filter(lambda elem: elem[0] != "module" and elem[0] != "cls", s.items())).items()
          }
          sensors.append(s_class(**s_args))

        drones.append(d_class(id = id, state = np.array(next(iter(drone["trajectory"].values()))["state"]), **d_args))
      self.__log_info(drones)

      for id, drone_traj in traj_dict.items():
        for time, state_meas_dict in drone_traj.items():
          drone = list(filter(lambda d: d.id == id, drones))[0]
          self.write_to_log(
            drone.generate_time_entry(
              np.array(state_meas_dict["state"]),
              [float(m["measurement"]) for m in state_meas_dict["measurements"]]
            ),
            int(time)
          )
      
    except FileNotFoundError as fnfe:
      print(fnfe)
      return False


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

  def get_drone_ids(self):
    return list(self.log["drones"].keys())

  def get_num_drone_sensors(self, drone_id):
    return len(self.log["drones"][drone_id]["info"].sensors)

  def get_drone_state_at_time(self, drone_id, timestep):
    return self.log["drones"][drone_id]["trajectory"][timestep].state
  
  def get_drone_sensor_measurements_at_time(self, drone_id, time_step):
    return [ms.measurement for ms in self.log["drones"][drone_id]["trajectory"][time_step].measurements]

  def get_drone_sensor_specs(self, drone_id, sensor_idx):
    return self.log["drones"][drone_id]["info"].sensors[sensor_idx]

  def get_traj_length(self):
    return len(self.log["drones"][self.get_drone_ids()[0]]["trajectory"])