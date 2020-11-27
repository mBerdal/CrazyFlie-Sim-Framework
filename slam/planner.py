import numpy as np
from slam.path_planning import AStar
import pyomo.environ as pe

class Planner:

    def __init__(self, **kwargs):
        self.min_len_frontier = kwargs.get("min_len_frontier", 10)
        self.padding_occ_grid = kwargs.get("padding_occ_grid", 3)
        self.max_range_observable = kwargs.get("max_range_observable",4)
        self.max_range_vis = kwargs.get("max_range_discount", 10)
        self.assignment_method = kwargs.get("assignment_method","optimized")

    def assign_waypoints(self, drones, shared_map):
        shared_map.compute_frontiers()
        shared_map.compute_occupancy_grid()
        points = shared_map.get_frontier_points(self.min_len_frontier)

        num_f = len(points)
        num_d = len(drones)
        print("Number of frontiers:", num_f)
        information_gain = []
        for i in range(len(points)):
            information_gain.append({"point": points[i], "information":
                len(shared_map.get_observable_cells_from_pos(points[i],max_range=self.max_range_observable))})

        dist = {}
        occ_grid = shared_map.get_occupancy_grid(pad=self.padding_occ_grid)
        for ind, p in enumerate(information_gain):
            dist_p = {}
            for d in drones.keys():
                cell_d = shared_map.cell_from_coordinate_local_map(d, drones[d])
                a = AStar(cell_d, p["point"], occ_grid)
                pad = self.padding_occ_grid-1
                while not a.init_sucsess and pad >= 0:
                    a = AStar(cell_d,p["point"],shared_map.get_occupancy_grid(pad=0))
                    pad -= 1
                res = a.planning()

                if res is None:
                    dist_p[d] = {"dist": np.inf, "wps": []}
                else:
                    dist_p[d] = {"dist": res[1]*shared_map.res, "wps": res[0]}
            dist[ind] = dist_p

        p_vis = np.zeros([num_f,num_f])
        for i in range(num_f):
            for j in range(i+1,num_f):
                p = self.compute_visibility_probability(points[i],points[j], shared_map)
                p_vis[i,j] = p
                p_vis[j,i] = p

        drone_ids = [k for k in drones.keys()]
        utility = np.zeros([num_f,num_d])
        for i in range(num_f):
            for j in range(num_d):
                utility[i, j] = self.utility_function(information_gain[i]["information"], dist[i][drone_ids[j]]["dist"])

        if self.assignment_method == "optimized":
            result = self.optimize_assignment(num_f, num_d, utility, p_vis)
        elif self.assignment_method == "greedy":
            result = self.greedy_assingment(num_f,num_d,utility,p_vis)
        else:
            distance = np.zeros([num_f,num_d])
            for i in range(num_f):
                for j in range(num_d):
                    distance[i,j] = dist[i][drone_ids[j]]["dist"]
            result = self.closest_assignment(num_f, num_d, distance)

        assignment = {}
        for j in range(num_d):
            if sum(result[:,j]) == 0:
                continue
            else:
                f = np.argmax(result[:,j])
                assignment[drone_ids[j]] = dist[f][drone_ids[j]]["wps"]
        return assignment

    def compute_visibility_probability(self, point1, point2, shared_map):
        if shared_map.check_collision(point1, point2):
            return 0
        dist = shared_map.res*np.linalg.norm(point1-point2)

        if dist > self.max_range_vis:
            return 0
        else:
            return 1 - dist/self.max_range_vis

    def utility_function(self, information, distance):
        return information/distance

    def optimize_assignment(self, num_f, num_d, utility, p_vis):

        model = pe.ConcreteModel()

        model.n = pe.RangeSet(1, num_f)
        model.m = pe.RangeSet(1, num_d)

        def utility_init(model, i, j):
            return utility[i-1,j-1]

        model.U = pe.Param(model.n, model.m, initialize=utility_init)

        def p_init(model, i, j):
            return p_vis[i-1,j-1]

        model.P = pe.Param(model.n, model.n, initialize=p_init)

        model.z = pe.Var(model.n, model.m, within=pe.Binary)
        model.D = pe.Var(model.n, within=pe.Reals)

        #Discount constraint

        def discount_rule(model, i):
            d = 0
            for k in model.n:
                d += model.P[i, k]*sum(model.z[k,:])
            return model.D[i] == (1-d)

        model.d_constraint = pe.Constraint(model.n, rule=discount_rule)

        def frontier_rule(model, i):
            return sum(model.z[i,:]) <= 1

        model.f_constraint = pe.Constraint(model.n, rule=frontier_rule)

        def drone_rule(model, j):
            return sum(model.z[:,j]) <= 1

        model.drone_constraint = pe.Constraint(model.m, rule=drone_rule)

        def objective(model):
            obj = 0
            for i in model.n:
                for j in model.m:
                   obj += model.D[i]*model.U[i,j]*model.z[i,j]
            return obj

        model.obj = pe.Objective(rule=objective,sense=pe.maximize)

        print("Optimizing Assignment")
        opt = pe.SolverFactory('gurobi',solver_io="python")
        opt.solve(model)

        assignment = np.zeros([num_f, num_d])
        for i in range(num_f):
            for j in range(num_d):
                assignment[i,j] = pe.value(model.z[i+1,j+1])
        return assignment

    def greedy_assingment(self, num_f, num_d, utility, p_vis):

        assignment = np.zeros([num_f, num_d])
        j = 0
        discount = np.ones([num_f,1])
        cur_utility = utility.copy()
        while j < num_d:
            ind = np.unravel_index(np.argmax(cur_utility),assignment.shape)
            assignment[ind[0],ind[1]] = 1
            utility[ind[0],:] = 0
            utility[:,ind[1]] = 0
            for i in range(num_f):
                discount[i, :] -= p_vis[i, ind[0]]
                cur_utility[i, :] = utility[i, :]*discount[i, :]
            j += 1
        return assignment

    def closest_assignment(self, num_f, num_d, dist):
        assignment = np.zeros([num_f, num_d])
        for j in range(num_d):
            ind = np.argmin(dist[:, j], axis=0)
            if dist[ind,j] == np.inf:
                continue
            assignment[ind, j] = 1
        return assignment
