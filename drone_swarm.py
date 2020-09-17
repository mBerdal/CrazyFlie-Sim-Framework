from environment import Environment
from utils.raytracing import multi_ray_intersect_triangle

import numpy as np

class DroneSwarm():
    def __init__(self, drones, controller):
        self.drones = drones
        self.controller = controller
        self.states = {drone.id: drone.state for drone in self.drones}
        min_ranges = []
        max_ranges = []
        for d in drones:
            ma, mi = d.get_sensor_limits()
            max_ranges.append(ma)
            min_ranges.append(mi)
        self.max_ragnes = np.concatenate(max_ranges)
        self.min_ranges = np.concatenate(min_ranges)

    def read_range_sensors(self,env: Environment):
        rays, orgins, idx_drones = self.get_range_rays()
        rays = rays.transpose()
        orgins = orgins.transpose()
        t_min = np.ones(rays.shape[0]) * np.inf
        for obj in env.get_objects():
            t = multi_ray_intersect_triangle(orgins, rays, obj["points"], 4)
            t_min = np.minimum(t_min, t)
        t_min[t_min < self.min_ranges] = self.min_ranges[t_min<self.min_ranges]
        drone_readings = {}
        for d in self.drones:
            local_readings = t_min[idx_drones[d.id]["start"]:idx_drones[d.id]["end"]]
            sensor_idx = d.get_sensor_idx()
            sensor_readings = []
            for idx in sensor_idx:
                sensor_readings.append(np.min(local_readings[idx["start"]:idx["end"]]))
            drone_readings[d.id] = sensor_readings
        return drone_readings

    def update_all_states(self,time_step):
      return {d.id: d.update_state(time_step) for d in self.drones}
    
    def update_all_commands(self, commands):
      for d in self.drones:
        d.update_command(commands[d.id])

    def sim_step(self, time_step, environment):
      sensor_readings = self.read_range_sensors(environment)
      self.update_all_commands(self.controller.get_commands(self.states, sensor_readings))
      drone_states = self.update_all_states(time_step)
      return drone_states, sensor_readings


    def get_range_rays(self):
        rays = []
        orgins = []
        idx_drones = {}
        idx = 0
        for d in self.drones:
            r, o, ma, mi = d.get_sensor_rays()
            rays.append(r)
            orgins.append(o)
            idx_drones[d.id] = {"start": idx, "end": idx + r.shape[1]}
            idx = idx + r.shape[1]
        rays = np.concatenate(rays, axis=1)
        orgins = np.concatenate(orgins, axis=1)
        return rays, orgins, idx_drones