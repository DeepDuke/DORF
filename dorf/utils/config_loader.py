#! /usr/bin/python2.7

import yaml
import math 


class Config:
    def __init__(self):
        ## coarse parameters ##
        self.image_w_reso = math.radians(1.0)  # default 1.0 dgeree
        self.image_h_reso = math.radians(1.0)
        self.local_map_radius = 60.0           # meters
        self.local_map_max_height = 5.0            # meters
        self.local_map_min_height = -3.0           # meters
        self.dyn_threshold = 1.0               # meters
        # LiDAR FOV in radians
        self.horizontal_fov = math.radians(360.0)
        self.theta_up = math.radians(2.0)
        self.theta_down = math.radians(-25.0)
        # receding horizon parameters
        self.max_window_size = 40
        self.sliding_window_size = 10
        self.ratio = 1.5
        self.beginning_ratio = 0.25
        self.end_ratio = 0.25
        # coarse reverting parameters
        self.revert_dist_threshold = 0.001
        self.neighbor_radius = 0.2
        
        ## Fine parameters ##
        self.ransac_height_threshold = 0.2        # Unit: meters
        self.z_threshold_alpha = 0.9
        self.distance_alpha = 2.0
        self.z_threshold_beta = 0.2
        self.distance_beta = 0.5
        
        self.n_ring = 60
        self.n_seg = 10
        
        ## Occupancy checking parameters ##
        self.ray_tracing_resolution = 1.0
        self.pure_static_threshold = 0.0165
        
        ## log path ##
        self.log_path = './log/'
        self.log_level = 'DEBUG'  # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        self.main_log_name = 'main.log'
        
        ## dataset path ##
        # rosbag 
        self.bag_path = {}
        # pcd
        self.pcd_path = {}
        
        ## save path ##
        self.result_path = './results/'
        
    def load(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Check if keys in config_dict are in self.__dict__
        for key in config_dict.keys():
            assert key in self.__dict__.keys(), 'Key {} not in Config class'.format(key)
        
        # Update the attributes of self
        for key, value in config_dict.items():
            setattr(self, key, value)
            
        # Update the derived attributes
        self.horizontal_fov = math.radians(self.horizontal_fov)
        self.theta_up = math.radians(self.theta_up)
        self.theta_down = math.radians(self.theta_down)
        self.vertical_fov = self.theta_up - self.theta_down
        self.image_w_reso = math.radians(self.image_w_reso)
        self.image_h_reso = math.radians(self.image_h_reso)
        
        self.range_image_width = int(self.horizontal_fov / self.image_w_reso)
        self.range_image_height = int(self.vertical_fov / self.image_h_reso)
        self.range_image_shape = (self.range_image_height, self.range_image_width)
        
        