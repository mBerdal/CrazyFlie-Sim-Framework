import numpy as np
from slam.path_planning import RRTStar

class Planner:

    def __init__(self, **kwargs):

        self.max_iter_rrt = kwargs.get("max_iter_rrt", 100)
        self.min_len_frontier = kwargs.get("min_len_frontier", 20)
        self.padding_occ_grid = kwargs.get("padding_occ_grid", 2)
        self.max_range = kwargs.get("max_range",4)

    def assign_waypoints_exploration(self, drones, shared_map):
        shared_map.compute_frontiers()
        shared_map.compute_occupancy_grid()
        points = shared_map.get_frontier_points(self.min_len_frontier)

        information_gain = []
        for i in range(len(points)):
            information_gain.append({"point": points[i], "observable_cells":
                shared_map.get_observable_cells_from_pos(points[i],max_range=self.max_range)})

        #Calculate distance from each drone to each possible frontier
        dist = {}
        free_cells = shared_map.get_free_cells()
        occ_grid = shared_map.get_occupancy_grid(pad=self.padding_occ_grid)
        for ind, p in enumerate(information_gain):
            dist_p = {}
            for d in drones.keys():
                cell_d = shared_map.cell_from_coordinate_local_map(d, drones[d])
                r = RRTStar(cell_d, p["point"], occ_grid, free_cells, run_to_max_iter=True, max_iter=self.max_iter_rrt)
                res = r.planning()
                if res is None:
                    dist_p[d] = {"dist": np.inf, "wps": []}
                else:
                    dist_p[d] = {"dist": res[1], "wps": res[0]}
            dist[ind] = dist_p

        drones_ids = [k for k in drones.keys()]
        assignment = {}

        information = []
        for ind in range(len(information_gain)):
            information_f = []
            for d in drones.keys():
                information_f.append(len(information_gain[ind]["observable_cells"])/dist[ind][d]["dist"])
            information.append(information_f)
        information = np.array(information)

        for i in range(min(len(drones), len(information_gain))):
            ind = np.unravel_index(np.argmax(information, axis=None), information.shape)
            assignment[drones_ids[ind[1]]] = dist[ind[0]][drones_ids[ind[1]]]["wps"]
            information[:,ind[1]] = 0
            information[ind[0],:] = 0
        return assignment
