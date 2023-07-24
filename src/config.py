#! /usr/bin/python2

import os
from math import pi 

# Constants
DEGREE_TO_RADIAN = pi / 180.0

## Coarse parameters
remo_params = {
    'IMAGE_W_RESO': 1.0 * DEGREE_TO_RADIAN,  # 1.0 dgeree
    'IMAGE_H_RESO': 1.0 * DEGREE_TO_RADIAN,  # 1.0 degree
    'LOCAL_MAP_RADIUS': 60.0,       # meters
    'LOCAL_MAP_MAX_HEIGHT': 5.0,    # meters
    'LOCAL_MAP_MIN_HEIGHT': -3.0,   # meters
    'DYN_THRESHOLD': 1.0,           # meters
    'HORIZONTAL_FOV': 2 * pi,       # 360 degrees
    'THETA_UP': 2 * DEGREE_TO_RADIAN,      # 2 degrees for semantic kitti dataset
    'THETA_DOWN': -25 * DEGREE_TO_RADIAN,  # -25 degrees for semantic kitti dataset
}

remo_params['VERTICAL_FOV'] = remo_params['THETA_UP'] - remo_params['THETA_DOWN']

remo_params['RANGE_IMAGE_WIDTH'] = int(remo_params['HORIZONTAL_FOV'] / remo_params['IMAGE_W_RESO'])
remo_params['RANGE_IMAGE_HEIGHT'] = int(remo_params['VERTICAL_FOV'] / remo_params['IMAGE_H_RESO']) 
remo_params['RANGE_IMAGE_SHAPE'] = (remo_params['RANGE_IMAGE_HEIGHT'], remo_params['RANGE_IMAGE_WIDTH'])

# Sliding window
remo_params['max_window_size'] = 40
remo_params['SLIDING_WINDOW_SIZE'] = 10
remo_params['RATIO'] = 1.5
remo_params['BEGINNING_RATIO'] = 0.25
remo_params['END_RATIO'] = 0.25

# Coarse Revert Threshold
remo_params['REVERT_DIST_THRESHOLD'] = 0.001 #0.005 #0.001
remo_params['NEIGHBOR_RADIUS'] = 0.2

## Fined parameters
remo_params['n_ring'] = 60
remo_params['n_seg'] = 10

remo_params['ray_tracing_resolution'] = 1.0 # meters
remo_params['pure_static_threshold'] = 0.0165 #0.016 #0.015 # meters

## Path
HOME = os.path.expandvars('$HOME') 

# Log path
log_path = HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/log'
log_level = 'DEBUG'  # (DEBUG, INFO, WARNING, ERROR, CRITICAL)
main_log_filename = 'remo.log'
corase_log_filename = 'coarse.log'
fined_log_filename = 'fined.log'
occupancy_log_filename = 'occupancy.log'

## Semantic Kitti Dataset Path

semantic_kitti_bag_path = {
    '00': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/00_4390_to_4530_w_interval_2_node.bag',
    '01': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/01_150_to_250_w_interval_1_node.bag',
    '02': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/02_860_to_950_w_interval_2_node.bag',
    '05': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/05_2350_to_2670_w_interval_2_node.bag',
    '07': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/07_630_to_820_w_interval_2_node.bag'
}

semantic_kitti_pcd_path = {
    '00': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/pcd/00_4390_to_4530_w_interval2_voxel_0.200000.pcd',
    '01': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/pcd/01_150_to_250_w_interval1_voxel_0.200000.pcd',
    '02': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/pcd/02_860_to_950_w_interval2_voxel_0.200000.pcd',
    '05': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/pcd/05_2350_to_2670_w_interval2_voxel_0.200000.pcd',
    '07': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/pcd/07_630_to_820_w_interval2_voxel_0.200000.pcd'
}

# Save map path
semantic_kitti_result_path = HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/results/'

## Gazebo Dataset 

gazebo_bag_path = {
    'ped_sample': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/bag/node_bag/node_ped_sample_voxel_0.2.bag',
    'ped_50': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/bag/node_bag/node_ped_50_voxel_0.2.bag',
    'ped_100': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/bag/node_bag/node_ped_100_voxel_0.2.bag',
    'ped_150': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/bag/node_bag/node_ped_150_voxel_0.2.bag',
}

gazebo_pcd_path = {
    'ped_sample': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/raw_pcd/ped_sample_raw_voxel_0.2.pcd',
    'ped_50': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/raw_pcd/ped_50_raw_voxel_0.2.pcd',
    'ped_100': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/raw_pcd/ped_100_raw_voxel_0.2.pcd',
    'ped_150': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/raw_pcd/ped_150_raw_voxel_0.2.pcd',
}

gazebo_result_path = HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/gazebo_data/results/'

remo_params['gazebo_sliding_window_size'] = 3
remo_params['gazebo_ratio'] = 1.5
remo_params['gazebo_beginning_ratio'] = 0.001
remo_params['gazebo_end_ratio'] = 0.001

remo_params['gazebo_theta_up'] = 22.5 * DEGREE_TO_RADIAN
remo_params['gazebo_theta_down'] = -22.5 * DEGREE_TO_RADIAN

remo_params['gazebo_vertical_fov'] = remo_params['gazebo_theta_up'] - remo_params['gazebo_theta_down']
remo_params['gazebo_horizontal_fov'] = 2 * pi

remo_params['gazebo_image_horizontal_reso'] = 1.5 * DEGREE_TO_RADIAN
remo_params['gazebo_image_vertical_reso'] = 1.5 * DEGREE_TO_RADIAN

remo_params['gazebo_range_image_width'] = int(remo_params['gazebo_horizontal_fov'] / remo_params['gazebo_image_horizontal_reso'])
remo_params['gazebo_range_image_height'] = int(remo_params['gazebo_vertical_fov'] / remo_params['gazebo_image_vertical_reso']) 
remo_params['gazebo_range_image_shape'] = (remo_params['gazebo_range_image_height'], remo_params['gazebo_range_image_width'])

remo_params['gazebo_pure_static_threshold'] = 0.015 #0.5 #0.016 #0.015 # meters

remo_params['gazebo_local_map_radius'] = 5.0        # meters
remo_params['gazebo_local_map_max_height'] = 5.0    # meters
remo_params['gazebo_local_map_min_height'] = -3.0   # meters
remo_params['gazebo_n_ring'] = 5
remo_params['gazebo_n_seg'] = 4


## UST datatset

ust_result_path = HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/results/'

ust_bag_path = {
    'ust_ped_1': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_ped_1_voxel_0.2.bag',
    'ust_ped_2': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_ped_2_voxel_0.2.bag',
    'ust_ped_3': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_ped_3_voxel_0.2.bag',
    'ust_ped_4': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_ped_4_voxel_0.2.bag',
    'ust_scene_2': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_scene_2_voxel_0.2.bag',
    'ust_scene_3': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_scene_3_voxel_0.2.bag',
    'ust_scene_4': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/bag/node_bag/node_ust_scene_4_voxel_0.2.bag'
}

ust_pcd_path = {
    'ust_ped_1': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_ped_1_raw.pcd',
    'ust_ped_2': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_ped_2_raw.pcd',
    'ust_ped_3': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_ped_3_raw.pcd',
    'ust_ped_4': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_ped_4_raw.pcd',
    'ust_scene_2': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_scene_2_raw.pcd',
    'ust_scene_3': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_scene_3_raw.pcd',
    'ust_scene_4': HOME + '/melodic_workspace/IROS2023/REMO_WS/src/REMO/hkust_data/raw_pcd/ust_scene_4_raw.pcd'
}

remo_params['ust_sliding_window_size'] = 10
remo_params['ust_ratio'] = 1.5
remo_params['ust_beginning_ratio'] = 0.001
remo_params['ust_end_ratio'] = 0.001

remo_params['ust_theta_up'] = 15 * DEGREE_TO_RADIAN
remo_params['ust_theta_down'] = -55 * DEGREE_TO_RADIAN

remo_params['ust_vertical_fov'] = remo_params['ust_theta_up'] - remo_params['ust_theta_down']
remo_params['ust_horizontal_fov'] = 2 * pi

remo_params['ust_image_horizontal_reso'] = 1.5 * DEGREE_TO_RADIAN
remo_params['ust_image_vertical_reso'] = 1.5 * DEGREE_TO_RADIAN

remo_params['ust_range_image_width'] = int(remo_params['ust_horizontal_fov'] / remo_params['ust_image_horizontal_reso'])
remo_params['ust_range_image_height'] = int(remo_params['ust_vertical_fov'] / remo_params['ust_image_vertical_reso']) 
remo_params['ust_range_image_shape'] = (remo_params['ust_range_image_height'], remo_params['ust_range_image_width'])

remo_params['ust_pure_static_threshold'] = 0.015 #0.5 #0.016 #0.015 # meters

remo_params['ust_local_map_radius'] = 30.0        # meters
remo_params['ust_local_map_max_height'] = 5.0    # meters
remo_params['ust_local_map_min_height'] = -3.0   # meters
remo_params['ust_n_ring'] = 60
remo_params['ust_n_seg'] = 10

