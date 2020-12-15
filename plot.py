from logger.read_log_maps import read_slam_log, read_shared_map_log
import matplotlib.pyplot as plt
import numpy as np

from utils.misc import compute_frontiers, compute_entropy_map, compute_information_map, compute_mean_information_map

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


def plot_last_maps(slam, shared_map=None, plot_trajectories=False, plot_loop=False, title=""):

    ids = [i for i in slam.keys()]

    for i in ids:
        plt.figure(i)
        t = slam[i]["t_maps"]
        map = slam[i]["maps"][t[-1]]
        res = slam[i]["info"]["res"]
        size_x = slam[i]["info"]["size_x"]
        size_y = slam[i]["info"]["size_y"]
        map = np.clip(map, -10, 10)
        map = np.exp(map) / (1 + np.exp(map))
        extent = [-size_x/2*res, size_x/2*res,-size_y/2*res, size_y/2*res]
        plt.imshow(map.T, "Greys", origin="lower",extent=extent)
        if plot_trajectories:
            t_poses = slam[i]["t_poses"]
            ind = t_poses.index(t[-1])
            t_poses = t_poses[0:ind]

            t_x = [slam[i]["poses"][j][0] for j in t_poses]
            t_y = [slam[i]["poses"][j][1] for j in t_poses]

            plt.plot(t_x, t_y)
        if plot_loop:
            t = [k for k in slam[i]["loop"].keys()]
            loop = slam[i]["loop"][t[-1]]

            for n in loop:
                for e in n[1]:
                    try:
                        plt.plot([n[0][0][0], loop[e][0][0][0]], [n[0][1][0], loop[e][0][1][0]], "-x",
                             color="red", markersize=4)
                    except:
                        continue
        plt.title(title)
    if shared_map is not None:
        plt.figure("SM")
        t = shared_map["timesteps"]
        map = shared_map["maps"][t[-1]]
        map = np.clip(map, -10, 10)
        map = np.exp(map)/(1+np.exp(map))
        res = shared_map["info"]["res"]
        size_x = shared_map["info"]["size_x"]
        size_y = shared_map["info"]["size_y"]
        extent = [-size_x / 2 * res, size_x / 2 * res, -size_y / 2 * res, size_y / 2 * res]
        plt.imshow(map.T,"Greys", origin="lower", extent=extent)

    plt.show()

def plot_last_shared_maps(shared_maps,slams,titles=[], max_t=[]):
    for i, s in enumerate(shared_maps):
        plt.figure()
        t = s["timesteps"]
        t_num = [float(q) for q in t]
        if max_t:
            t = [a for a,b in zip(t,t_num) if b <= max_t[i]]
        map = s["maps"][t[-1]]
        map = np.clip(map, -10, 10)
        map = np.exp(map)/(1+np.exp(map))
        res = s["info"]["res"]
        size_x = s["info"]["size_x"]
        size_y = s["info"]["size_y"]
        extent = [-size_x / 2 * res, size_x / 2 * res, -size_y / 2 * res, size_y / 2 * res]
        plt.imshow(map.T,"Greys", origin="lower", extent=extent)
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        for p in slams[i].keys():
            t_poses = slams[i][p]["t_poses"]
            t_poses_num = [float(a) for a in t_poses]
            t_poses = [a for a,b in zip(t_poses,t_poses_num) if b <= float(t[-1])]
            #ind = t_poses.index(t[-1])
            #t_poses = t_poses[0:ind]

            t_x = [slams[i][p]["poses"][j][0] for j in t_poses]
            t_y = [slams[i][p]["poses"][j][1] for j in t_poses]
            plt.plot(t_x, t_y)
        legend = ["Robot: {}".format(p) for p in slams[i].keys()]
        plt.legend(legend)
        if titles:
            plt.title(titles[i])
    plt.show()

def plot_entropy_maps(slam, shared_map=None):
    ids = [i for i in slam.keys()]
    plt.figure()
    for i in ids:
        entropy = []
        for t in slam[i]["t_maps"]:
            map = slam[i]["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            entropy.append(compute_entropy_map(map))
        time = [float(t) for t in slam[i]["t_maps"]]
        plt.plot(time,entropy)
    plt.legend(ids)
    plt.grid()
    plt.show()

def plot_information_maps(slam, shared_map=None):
    ids = [i for i in slam.keys()]
    plt.figure()
    for i in ids:
        entropy = []
        for t in slam[i]["t_maps"]:
            map = slam[i]["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            entropy.append(compute_information_map(map))
        time = [float(t) for t in slam[i]["t_maps"]]
        plt.plot(time, entropy)
    plt.legend(ids)
    plt.grid()
    plt.show()

def plot_mean_information_maps(slam,shared_map=None):
    ids = [i for i in slam.keys()]
    plt.figure()
    for i in ids:
        entropy = []
        for t in slam[i]["t_maps"]:
            map = slam[i]["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            entropy.append(compute_mean_information_map(map))
        time = [float(t) for t in slam[i]["t_maps"]]
        plt.plot(time, entropy)
    plt.legend(ids)
    plt.grid()
    plt.show()




def plot_frontiers(shared_map, starting_cell,l=10):
    timesteps = shared_map["timesteps"]
    res = shared_map["info"]["res"]
    size_x = shared_map["info"]["size_x"]
    size_y = shared_map["info"]["size_y"]
    extent = [-size_x / 2 * res, size_x / 2 * res, -size_y / 2 * res, size_y / 2 * res]
    for t in timesteps[0:l]:
        plt.figure()
        map = shared_map["maps"][t]
        frontiers = compute_frontiers(map, starting_cell,0.36)
        map = np.clip(map, -10, 10)
        map = np.exp(map) / (1 + np.exp(map))
        plt.imshow(map.T, "Greys",origin="lower")
        for front in frontiers:
            if len(front) < 5:
                continue
            plt.plot([c[0] for c in front],[c[1] for c in front],"o",markersize=1.5)
    plt.show()

def compare_entropy(shared_maps):
    plt.figure()
    for s in shared_maps:
        entropy = []
        for t in s["t_maps"]:
            map = s["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            entropy.append(compute_mean_information_map(map))
        time = [float(t) for t in s["t_maps"]]
        plt.plot(time, entropy)
    plt.legend


def compare_mean_information_sm_s(shared_map, slam, max_t=None, legend=[]):
    entropy = []
    plt.figure()
    for t in shared_map["timesteps"]:
        map = shared_map["maps"][t]
        map = np.clip(map, -10, 10)
        map = np.exp(map) / (1 + np.exp(map))
        if max_t:
            if float(t) <= max_t[i]:
                entropy.append(compute_mean_information_map(map))
        else:
            entropy.append(compute_mean_information_map(map))
        if max_t:
            time = [float(t) for t in shared_map["timesteps"] if float(t) <= max_t]
        else:
            time = [float(t) for t in shared_map["timesteps"]]
    plt.plot(time, entropy)
    for i in slam.keys():
        entropy = []
        for t in slam[i]["t_maps"]:
            map = slam[i]["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            if max_t:
                if float(t) <= max_t[i]:
                    entropy.append(compute_mean_information_map(map))
            else:
                entropy.append(compute_mean_information_map(map))
        if max_t:
            time = [float(t) for t in slam[i]["t_maps"] if float(t) <= max_t]
        else:
            time = [float(t) for t in slam[i]["t_maps"]]
        plt.plot(time,entropy)
    plt.legend(legend)
    plt.grid()
    plt.title("Comparison Mean Information")
    plt.xlabel("Time [s]")
    plt.ylabel("Mean Information")
    plt.show()



def plot_information_sm_maps(shared_maps,legend=[], max_t=[]):
    plt.figure()
    for i,s in enumerate(shared_maps):
        entropy = []
        for t in s["timesteps"]:
            map = s["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            if max_t:
                if float(t) <= max_t[i]:
                    entropy.append(compute_information_map(map))
            else:
                entropy.append(compute_information_map(map))
        if max_t:
            time = [float(t) for t in s["timesteps"] if float(t) <= max_t[i]]
        else:
            time = [float(t) for t in s["timesteps"]]
        plt.plot(time, entropy)
    plt.legend(legend)
    plt.xlabel("Time [s]")
    plt.ylabel("Information")
    plt.title("Comparison Information")
    plt.grid()
    plt.figure()
    for i,s in enumerate(shared_maps):
        entropy = []
        for t in s["timesteps"]:
            map = s["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            if max_t:
                if float(t) <= max_t[i]:
                    entropy.append(compute_mean_information_map(map))
            else:
                entropy.append(compute_mean_information_map(map))
        if max_t:
            time = [float(t) for t in s["timesteps"] if float(t) <= max_t[i]]
        else:
            time = [float(t) for t in s["timesteps"]]
        plt.plot(time, entropy)
    plt.legend(legend)
    plt.xlabel("Time [s]")
    plt.ylabel("Mean Information")
    plt.title("Comparison Mean Information")
    plt.grid()
    plt.figure()
    for i,s in enumerate(shared_maps):
        entropy = []
        for t in s["timesteps"]:
            map = s["maps"][t]
            map = np.clip(map, -10, 10)
            map = np.exp(map) / (1 + np.exp(map))
            if max_t:
                if float(t) <= max_t[i]:
                    entropy.append(compute_entropy_map(map))
            else:
                entropy.append(compute_entropy_map(map))
        if max_t:
            time = [float(t) for t in s["timesteps"] if float(t) <= max_t[i]]
        else:
            time = [float(t) for t in s["timesteps"]]
        plt.plot(time, entropy)
    plt.legend(legend)
    plt.xlabel("Time [s]")
    plt.ylabel("Entropy")
    plt.title("Comparison Entropy")
    plt.grid()

    plt.show()


def loop_closure_comparison():
    shared_name_1 = "log_backup/1_drone_loop_low_closure/shared_map21.json"
    slam_name_1 = "log_backup/1_drone_loop_low_closure/slam21.json"

    shared_name_2 = "log_backup/1_drone_loop_high_closure/shared_map25.json"
    slam_name_2 = "log_backup/1_drone_loop_high_closure/slam25.json"

    sm = []
    names = [shared_name_1, shared_name_2]
    for n in names:
        sm.append(read_shared_map_log(n))

    plot_information_sm_maps(sm, legend=["Low Loop Closure", "High Loop Closure"])

    names = [slam_name_1, slam_name_2]
    slams = []
    for n in names:
        slams.append(read_slam_log(n))
    plot_last_maps(slams[0], shared_map=None, plot_trajectories=True, title="Low Loop Closure")
    plot_last_maps(slams[1], shared_map=None, plot_trajectories=True, title="High Loop Closure")


def different_number_comaprison():
    sh_map_1 = "log_backup/1_drone_loop_low_closure/shared_map34.json"
    sh_map_2 = "log_backup/2_drone_loop_low_closure/shared_map27.json"
    sh_map_3 = "log_backup/3_drone_loop_low_closure/shared_map33.json"
    sh_map_4 = "log_backup/4_drone_loop_low_closure/shared_map32.json"

    slam_1 = "log_backup/1_drone_loop_low_closure/slam34.json"
    slam_2 = "log_backup/2_drone_loop_low_closure/slam27.json"
    slam_3 = "log_backup/3_drone_loop_low_closure/slam33.json"
    slam_4 = "log_backup/4_drone_loop_low_closure/slam32.json"

    shared_names = [sh_map_1, sh_map_2, sh_map_3, sh_map_4]
    slams_names = [slam_1, slam_2, slam_3, slam_4]

    shared_maps = []
    for n in shared_names:
        shared_maps.append(read_shared_map_log(n))

    slams = []
    for n in slams_names:
        slams.append(read_slam_log(n))

    titles = ["Exploration using 1 robot", "Exploration using 2 robots","Exploration using 3 robots", "Exploration using 4 robots"]
    max_t = [170, 100, 80, 70]
    plot_last_shared_maps(shared_maps, slams, titles=titles, max_t=max_t)
    legend = ["Num Robot(s): 1","Num Robot(s): 2","Num Robot(s): 3","Num Robot(s): 4"]
    plot_information_sm_maps(shared_maps,legend=legend,max_t=max_t)

def mean_information_comp():
    sh_map_4 = "log_backup/4_drone_loop_low_closure/shared_map30.json"
    slam_4 = "log_backup/4_drone_loop_low_closure/slam30.json"
    sm = read_shared_map_log(sh_map_4)
    slam = read_slam_log(slam_4)

    compare_mean_information_sm_s(sm, slam, legend=["Shared Map", "Map 0","Map 1","Map 2","Map 3"])

def office_last_maps():
    sh_1 = "log_backup/3_drone_office_noisy/shared_map42.json"
    sh_2 = "log_backup/3_drone_office_noisy/shared_map49.json"
    sh_3 = "log_backup/3_drone_office_noisy/shared_map48.json"

    sh_name = [sh_1, sh_2, sh_3]
    shared_maps = []
    for n in sh_name:
        shared_maps.append(read_shared_map_log(n))

    slam_1 = "log_backup/3_drone_office_noisy/slam42.json"
    slam_2 = "log_backup/3_drone_office_noisy/slam49.json"
    slam_3 = "log_backup/3_drone_office_noisy/slam48.json"
    slam_name = [slam_1, slam_2, slam_3]
    slams = []
    for n in slam_name:
        slams.append(read_slam_log(n))

    titles = ["Exploration Middle", "Exploration Upper Left ", "Exploration Lower Right"]
    max_t = [120, 140, 140]
    plot_last_shared_maps(shared_maps, slams, titles=titles, max_t=max_t)
    legend = ["Middle", "Upper Left", "Lower Right"]
    plot_information_sm_maps(shared_maps, legend=legend, max_t=max_t)

if __name__ == "__main__":
    office_last_maps()
    #mean_information_comp()
    #different_number_comaprison()
    #loop_closure_comparison()
    #slam_file = read_slam_log(slam_name)
    #shared_file = read_shared_map_log(shared_name)

    #plot_last_maps(slam_file, shared_map=shared_file, plot_trajectories=True,plot_loop=False)
    #plot_entropy_maps(slam_file)
    #plot_information_maps(slam_file)
    #plot_mean_information_maps(slam_file)
    #plot_frontiers(shared_name, [200,200])