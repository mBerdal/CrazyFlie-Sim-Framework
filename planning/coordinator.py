import numpy as np
from planning.path_planning import AStar
from utils.misc import compute_entropy_map
import pyomo.environ as pe

class Coordinator:
    """
    Class for coordinating the drones based on frontier exploration and active loop closing.

    The coordination is done using an information based optimization procedure. The optimization can be solved using
    Gurobi or a greedy approach.
    """
    def __init__(self, **kwargs):
        self.min_len_frontier = kwargs.get("min_len_frontier", 20)
        self.padding_occ_grid = kwargs.get("padding_occ_grid", 4)
        self.max_range_observable = kwargs.get("max_range_observable", 4)
        self.max_range_vis = kwargs.get("max_range_discount", 10)
        self.assignment_method = kwargs.get("assignment_method","optimized")

        self.p_free = kwargs.get("p_free", 0.4)
        self.p_occupied = kwargs.get("p_occupied", 0.7)

        self.log_free = np.log(self.p_free / (1 - self.p_free))
        self.log_occupied = np.log(self.p_occupied / (1 - self.p_occupied))

        self.loop_gain = kwargs.get("loop_gain", 30)

    def assign_waypoints(self, drones, shared_map, loop_closures, compare=False):
        #Extract all possible frontiers based on global map
        shared_map.compute_frontiers()
        shared_map.compute_occupancy_grid()
        points = shared_map.get_frontier_points(self.min_len_frontier)

        num_f = len(points)
        num_d = len(drones)
        num_l = sum([len(loop_closures[k]) for k in loop_closures.keys()])
        print("Number of frontiers:", num_f)
        print("Number of loop closures:", num_l)
        information_gain = []
        prob_map = shared_map.convert_grid_to_prob()
        original_entropy = compute_entropy_map(prob_map)

        #Stack all possible loop closures from all drones into a single array
        loop_c = []
        for k in loop_closures.keys():
            for l in loop_closures[k]:
                tmp = [l[i] for i, _ in enumerate(l)]
                tmp[0] = shared_map.cell_from_coordinate_local_map(k, l[0])
                loop_c.append(tmp)

        #Compute the reduction in entropy from observing at each possible frontier
        for i in range(len(points)):
            observable_cells, occupied_cells = shared_map.get_observable_cells_from_pos(points[i],max_range=self.max_range_observable)
            updated_map = self.update_map(shared_map.get_map(), observable_cells, occupied_cells)
            exp = np.exp(np.clip(updated_map,-10,10))
            prob_map = exp/(1 + exp)
            updated_entropy = compute_entropy_map(prob_map)
            information_gain.append({"point": points[i], "information": original_entropy - updated_entropy})

        #Compute the distance from all drones to all frontiers
        dist = {}
        occ_grid = shared_map.get_occupancy_grid(pad=self.padding_occ_grid)
        for ind, p in enumerate(information_gain):
            dist_p = {}
            for d in drones.keys():
                cell_d = shared_map.cell_from_coordinate_local_map(d, drones[d])
                a = AStar(cell_d, p["point"], occ_grid)
                pad = self.padding_occ_grid-1
                while not a.init_sucsess and pad >= 0:
                    a = AStar(cell_d, p["point"], shared_map.get_occupancy_grid(pad=pad))
                    pad -= 1
                res = a.planning()

                if res is None:
                    dist_p[d] = {"dist": np.inf, "wps": []}
                else:
                    dist_p[d] = {"dist": res[1]*shared_map.res, "wps": res[0]}
            dist[ind] = dist_p

        #Compute the probability of visibility between all pairs of frontiers
        p_vis = np.zeros([num_f+num_l,num_f+num_l])
        for i in range(num_f):
            for j in range(i+1,num_f):
                p = self.compute_visibility_probability(points[i],points[j], shared_map)
                p_vis[i,j] = p
                p_vis[j,i] = p

        #Compute the utility of assigning frontiers to each drone
        drone_ids = [k for k in drones.keys()]
        utility = np.zeros([num_f+num_l,num_d])
        for i in range(num_f):
            for j in range(num_d):
                utility[i, j] = self.utility_function(information_gain[i]["information"], dist[i][drone_ids[j]]["dist"])
        #Compute the utility of performing loop closure
        i = num_f
        for k in loop_closures.keys():
            for l in loop_closures[k]:
                utility[i, k] = self.utility_loop_closure(l[1], l[2])
                i += 1

        invalid_indx = (np.isinf(utility) | np.isnan(utility))
        utility[invalid_indx] = 0

        print(utility)

        if compare:
            opt_res = self.optimize_assignment(num_f+num_l, num_d, utility, p_vis)
            greedy_res = self.greedy_assingment(num_f+num_l, num_d, utility.copy(), p_vis)
            score_greedy = self.score_assingment(greedy_res, utility, p_vis)
            score_opt = self.score_assingment(opt_res, utility, p_vis)
            distance = np.zeros([num_f, num_d])
            for i in range(num_f):
                for j in range(num_d):
                    distance[i, j] = dist[i][drone_ids[j]]["dist"]
            closest_res = self.closest_assignment(num_f, num_d,distance)
            closest_score = self.score_assingment(closest_res, utility, p_vis)
            return opt_res, greedy_res, closest_res, score_opt, score_greedy, closest_score

        #Find the assignment of each drone based on the desired solver
        if self.assignment_method == "optimized":
            result = self.optimize_assignment(num_f+num_l, num_d, utility, p_vis)
        elif self.assignment_method == "greedy":
            result = self.greedy_assingment(num_f+num_l,num_d,utility,p_vis)
        else:
            distance = np.zeros([num_f,num_d])
            for i in range(num_f):
                for j in range(num_d):
                    distance[i,j] = dist[i][drone_ids[j]]["dist"]
            result = self.closest_assignment(num_f, num_d, distance)

        #Return the waypoints to the drones
        assignment = {}
        for j in range(num_d):
            if sum(result[:,j]) == 0:
                continue
            else:
                f = np.argmax(result[:,j])
                if f < num_f:
                    assignment[drone_ids[j]] = (dist[f][drone_ids[j]]["wps"], "active")
                else:
                    assignment[drone_ids[j]] = ([loop_c[f-num_f][0]], "loop_closing")
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

    def utility_loop_closure(self, d_dist, t_dist):
        return t_dist/d_dist*self.loop_gain

    def optimize_assignment(self, num_t, num_d, utility, p_vis):

        model = pe.ConcreteModel()

        model.n = pe.RangeSet(1, num_t)
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

        assignment = np.zeros([num_t, num_d])
        for i in range(num_t):
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
            dist[ind,:] = np.inf
        return assignment

    def score_assingment(self, assignment, utility, p_vis):
        num_f = p_vis.shape[0]
        num_d = utility.shape[1]
        D = np.zeros([p_vis.shape[0],1])
        ass = np.sum(assignment,axis=1)
        for i in range(num_f):
            D[i] = 1 - np.sum(p_vis[i,:]*ass)
        score = 0
        for i in range(num_f):
            for j in range(num_d):
                score += D[i]*utility[i,j]*assignment[i,j]
        return score

    def update_map(self, map, observable_cells, occupied_cells):
        for c in observable_cells:
            if not (c[0] < 0 or c[0] >= map.shape[0] or c[1] < 0 or c[1] >= map.shape[1]):
                map[c[0],c[1]] += self.log_free
        for c in occupied_cells:
            if not(c[0] < 0 or c[0] >= map.shape[0] or c[1] < 0 or c[1] >= map.shape[1]):
                map[c[0],c[1]] += self.log_occupied
        return map

