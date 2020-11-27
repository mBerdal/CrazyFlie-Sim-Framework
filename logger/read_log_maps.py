from utils.json_utils import read_json
import numpy as np

def read_shared_map_log(filename):
    data = read_json(filename)
    timesteps = [k for k in data["log"].keys()]

    map_info = {}
    map_info["res"] = data["info"]["map_res"]
    map_info["size_x"] = data["info"]["map_size_x"]
    map_info["size_y"] = data["info"]["map_size_y"]

    maps = {}
    for t in timesteps:
        try:
            maps[t] = np.array(data["log"][t]["map"])
        except KeyError:
            pass
    timesteps = [k for k in maps.keys()]

    return {"maps": maps, "info": map_info, "timesteps": timesteps}


def read_slam_log(filename):
    data = read_json(filename)
    ids = [k for k in data.keys()]
    slams = {}
    for id in ids:
        timesteps = [k for k in data[id]["log"].keys()]

        slam_info = {}
        slam_info["res"] = data[id]["info"]["map_res"]
        slam_info["size_x"] = data[id]["info"]["map_size_x"]
        slam_info["size_y"] = data[id]["info"]["map_size_y"]
        slam_info["num_particles"] = data[id]["info"]["num_particles"]


        maps = {}
        for t in timesteps:
            try:
                maps[t] = np.array(data[id]["log"][t]["map"])
            except KeyError:
                pass
        t_maps = [k for k in maps.keys()]

        poses = {}
        for t in timesteps:
            try:
                poses[t] = np.array(data[id]["log"][t]["pose"]).reshape(3,1)
            except KeyError:
                pass
        t_poses = [k for k in poses.keys()]
        slams[id] = {"info": slam_info, "poses": poses, "t_poses": t_poses, "maps": maps, "t_maps": t_maps}
    return slams


def read_control_log(filename):
    data = read_json(filename)
    ids = [k for k in data.keys()]
    waypoints = {}
    for id in ids:
        timesteps = [k for k in data[id]["log"]]

        wps = {}
        for t in timesteps:
            try:
                w = data[id]["log"][t]["waypoints"]
                if w:
                    cur = data[id]["log"][t]["current_waypoint"]
                    wp = [np.array(w[i]) for i in range(cur,len(w))]
                    wps[t] = wp
                else:
                    continue
            except KeyError:
                pass
        waypoints[id] = wps

    return waypoints