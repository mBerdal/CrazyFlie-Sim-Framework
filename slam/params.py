import numpy as np

slam_params = {
    "threshold_resampling": 2
}

particle_params = {
}

map_params = {
    "size_x": 500,
    "size_y": 500,
    "res": 0.10,
    "max_range": 10,
}

scan_match_params = {
        "max_iterations": 5,
        "delta_x": 0.05,
        "delta_y": 0.05,
        "delta_theta": 0.05,
        "step": 1,
        "sigma": 0.1,
        "max_sensor_range": 10,
        "skip_rays": 1,
    }

odometry_params = {
    "alpha1": 0.15,
    "alpha2": 0.10,
    "alpha3": 0.25,
    "alpha4": 0.25
}

obs_params = {
    "sigma": 0.10,
    "occ_threshold": 0.7,
    "max_sensor_range": 10,
    "skip_rays": 1
}