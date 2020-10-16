from environment.environment import Environment
from environment.obstacle import Obstacle

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

"""

  def __init__(self, drones = [], environment = None):
    self.log = {"environment": environment, "drones": {}}
    self.__log_info(drones)
      
  def load_from_file(self, filename: str = ""):
    try:
      data = read_json(filename)

      drones_dict = data.get("drones", None)
      assert not drones_dict is None, f"Drones not specified in file {filename}. Failed loading from file."

      env_dict = data.get("environment", None)
      assert not env_dict is None, f"Environment not specified in file {filename}. Failed loading from file."

      assert "trajectory" in dict(next(iter(drones_dict.values()))).keys(), f"Trajectories not specified in file {filename}. Failed loading from file."
      traj_dict = {id: drone_dict["trajectory"] for id, drone_dict in drones_dict.items()}

      if "path" in env_dict.keys():
        env_dict = read_json(env_dict["path"])
      self.log["environment"] =  Environment([Obstacle(obj["shape"], np.array(obj["points"])) for obj in env_dict["obstacles"]])

      drones = []
      for id, drone in drones_dict.items():
        d_class = getattr(importlib.import_module(drone["info"]["module"]), drone["info"]["cls"])
        d_args = dict(filter(lambda elem: elem[0] != "module" and elem[0] != "cls" and elem[0] != "id", drone["info"].items()))

        sensors = []
        for s in drone["info"]["sensors"]:
          s_class = getattr(importlib.import_module(s["module"]), s["cls"])
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
          self.log_time_step(
            drone.generate_time_entry(
              np.array(state_meas_dict["state"]),
              [np.array(m["measurement"]) for m in state_meas_dict["measurements"]]
            ),
            float(time)
          )
      
    except FileNotFoundError:
      print(f"Could not find file named {filename}. Loading log file failed. Exiting.")
      exit(0)

  def log_time_step(self, log_entry, timestep) -> None:
    try:
      self.log["drones"][log_entry.id]["trajectory"][timestep] = log_entry
    except KeyError:
      self.log["drones"][log_entry.id]["trajectory"] = {timestep: log_entry}

  def log_info(self, log_entry) -> None:
    self.log["drones"][log_entry.id]["info"] = log_entry

  def save_log(self, filename, env_to_file = None):
    assert not filename is None, "filename not supplied. save_log failed"

    lg = deepcopy(self.log)
    for _, drone_log in lg["drones"].items():
      drone_log["info"] = drone_log["info"].to_JSONable()
      for step, step_log in drone_log["trajectory"].items():
        drone_log["trajectory"][step] = step_log.to_JSONable()

    if isinstance(self.log["environment"], str):
      lg["environment"] = {"path": self.log["environment"]}
    else:
      if env_to_file is None:
        lg["environment"] = self.log["environment"].to_JSONable()
      else:
        lg["environment"] = {"path": env_to_file}
        write_json(env_to_file, self.log["environment"].to_JSONable())

    write_json(filename, lg)

  def get_environment(self):
    return self.log["environment"]

  def get_drone_ids(self):
    return list(self.log["drones"].keys())

  def get_num_drone_sensors(self, drone_id):
    return len(self.log["drones"][drone_id]["info"].sensors)

  def get_log_end_time(self):
    return float(max(self.log["drones"][self.get_drone_ids()[0]]["trajectory"].keys()))

  def get_log_step_length(self):
    it = iter(self.log["drones"][self.get_drone_ids()[0]]["trajectory"].keys())
    return -next(it) + next(it)

  def get_sensor_class(self, drone_id, sensor_idx):
    return getattr(
      importlib.import_module(
        self.log["drones"][drone_id]["info"].sensors[sensor_idx].module
      ),
      self.log["drones"][drone_id]["info"].sensors[sensor_idx].cls
    )

  def get_drone_class(self, drone_id):
    return getattr(
      importlib.import_module(
        self.log["drones"][drone_id]["info"].module
      ),
      self.log["drones"][drone_id]["info"].cls
    )

  def get_sensor_plotting_kwargs(self, drone_id, sensor_idx, time):
    return {
      **{"state": self.__get_drone_state_at_time(drone_id, time)},
      **dict(filter(lambda elem: elem[0] != "cls" and elem[0] != "module", self.__get_drone_sensor_specs(drone_id, sensor_idx).__dict__.items())),
      **{"measurement": self.__get_drone_sensor_measurements_at_time(drone_id, sensor_idx, time)}
    }

  def get_drone_plotting_kwargs(self, drone_id, time):
    return {"state": self.__get_drone_state_at_time(drone_id, time)}
    
  def __log_info(self, drones):
    for d in drones:
      self.log["drones"][d.id] = {"info": d.get_info_entry()}

  def __get_drone_state_at_time(self, drone_id, timestep):
    return self.log["drones"][drone_id]["trajectory"][timestep].state
  
  def __get_drone_sensor_measurements_at_time(self, drone_id, sensor_idx, time_step):
    return [ms.measurement for ms in self.log["drones"][drone_id]["trajectory"][time_step].measurements][sensor_idx]

  def __get_drone_sensor_specs(self, drone_id, sensor_idx):
    return self.log["drones"][drone_id]["info"].sensors[sensor_idx]

  def get_drone_sensor_measurements(self,drone_id,sensor_idx):
    measurements = {}
    step_length = self.get_log_step_length()
    end_time = self.get_log_end_time()
    for i in range(np.int(end_time/step_length)):
        measurements[i] =  self.__get_drone_sensor_measurements_at_time(drone_id,sensor_idx,i*step_length)
    return {"step_length": step_length, "steps": i, "measurements": measurements}

  def get_drone_states(self,drone_id):
    states = {}
    step_length = self.get_log_step_length()
    end_time = self.get_log_end_time()
    for i in range(np.int(end_time/step_length)):
      states[i] = self.__get_drone_state_at_time(drone_id,i*step_length)
    return {"step_length": step_length,"steps": i, "states": states}

  def get_drone_sensor_specs(self,drone_id, sensor_idx):
    return self.__get_drone_sensor_specs(drone_id,sensor_idx)