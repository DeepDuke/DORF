#! /usr/bin/python2.7

import numpy as np
import multiprocessing as mp
from collections import Counter
from functools import partial
from math import pi, sqrt, atan2
from sklearn.linear_model import RANSACRegressor

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import  PointField
from std_msgs.msg import Header

from pypcd import pypcd 

from dorf.utils.range_image_utils import transform_local_map_to_lidar_frame, transform_local_map_to_lidar_frame_gazebo
from dorf.utils.color_utils import get_ground_object_color, get_static_object_color


class Bin:
    def __init__(self, min_r=0.0, max_r=0.0, min_anlge=0.0, max_angle=0.0):
        self.min_r = min_r
        self.max_r = max_r
        self.min_angle = min_anlge
        self.max_angle = max_angle
        self.point_ids = []
        self.ground_point_ids = []

def save_ground_static_map(ground_point_raw_ids, pcd_raw_points, save_path):
    print('There are {} / {} ground points detected'.format(len(ground_point_raw_ids), len(pcd_raw_points)))
    # Convert static map points to PointCloud2 msg
    points = np.array([[pt[0], pt[1], pt[2], pt[3], get_static_object_color()] for pt in pcd_raw_points])
    for id in ground_point_raw_ids:
        points[id, 4] = get_ground_object_color()
   
    fields=[PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.UINT32, 1),
            PointField('rgba', 16, PointField.UINT32, 1)]

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'map'
    pc2_msg = point_cloud2.create_cloud(header, fields, points)
    
    # Convert pc2_msg to pcd
    obj = pypcd.PointCloud.from_msg(pc2_msg)
    obj.save(save_path)

def fine_filter(pcd_raw_points, pcd_kdtree, node_msg_list,RESULT_SAVING_PATH, args, config):
    
    ground_point_raw_ids, freq_dict, total_static_point_raw_ids, total_dynamic_point_raw_ids = ground_segmentation_mp(pcd_raw_points, pcd_kdtree, node_msg_list, args, config)
    
    # Save map
    save_ground_static_map(ground_point_raw_ids, pcd_raw_points, RESULT_SAVING_PATH['ground_static_map_path'])
    
    return ground_point_raw_ids, freq_dict, total_static_point_raw_ids, total_dynamic_point_raw_ids

def ground_segmentation_mp(pcd_raw_points, pcd_kdtree, node_msg_list, args, config):
    result = mp.Pool(20).map(partial(scan_ratio_test, pcd_raw_points=pcd_raw_points, pcd_kdtree=pcd_kdtree, args=args, config=config), node_msg_list)
    
    total_ground_point_raw_ids = []
    total_static_point_raw_ids = []
    total_dynamic_point_raw_ids = []
    for node_res in result:
        submap_ground_raw_ids, submap_static_raw_ids, submap_dynamic_raw_ids = node_res
        total_ground_point_raw_ids.extend(submap_ground_raw_ids)
        total_static_point_raw_ids.extend(submap_static_raw_ids)
        total_dynamic_point_raw_ids.extend(submap_dynamic_raw_ids)

    # count the id frequency, only those with higher
    counter = Counter(total_ground_point_raw_ids)
    freq_dict = dict(counter)

    total_ground_point_raw_ids = list(set(total_ground_point_raw_ids))
    total_static_point_raw_ids = list(set(total_static_point_raw_ids))
    total_dynamic_point_raw_ids = list(set(total_dynamic_point_raw_ids))

    return total_ground_point_raw_ids, freq_dict, total_static_point_raw_ids, total_dynamic_point_raw_ids

def scan_ratio_test(node_msg, pcd_raw_points, pcd_kdtree, args, config):
    # Get scan bin list
    scan_VOI_pts = get_scan_VOI_info(node_msg, args, config)
    scan_bin_list = get_scan_VOI_bin_list(scan_VOI_pts, args, config)
    
    # Get submap bin list
    SOI_pts, SOI_pt_raw_ids = get_SOI_info(node_msg, pcd_raw_points, pcd_kdtree, args, config)
    submap_bin_list  = get_basic_bin_list(SOI_pts, args, config)
    
    # scan ratio checking
    result = map(partial(single_bin_scan_ratio_test, scan_VOI_pts=scan_VOI_pts, SOI_pts=SOI_pts, SOI_pt_raw_ids=SOI_pt_raw_ids, args=args, config=config), list(zip(scan_bin_list, submap_bin_list)))
    
    submap_ground_raw_ids = []
    submap_static_raw_ids = []
    submap_dynamic_raw_ids = []
    for ground_raw_ids, static_raw_ids, dynamic_raw_ids in result:
        submap_ground_raw_ids.extend(ground_raw_ids)
        submap_static_raw_ids.extend(static_raw_ids)
        submap_dynamic_raw_ids.extend(dynamic_raw_ids)
    
    return submap_ground_raw_ids, submap_static_raw_ids, submap_dynamic_raw_ids  

def get_scan_VOI_info(node_msg, args, config):
    LOCAL_MAP_RADIUS = config.local_map_radius
    LOCAL_MAP_MIN_HEIGHT = config.local_map_min_height
    LOCAL_MAP_MAX_HEIGHT = config.local_map_max_height
    
    _, node, _ = node_msg
    pc_msg = node.lidar
    scan_points = point_cloud2.read_points_list(pc_msg)  # N * 4, each is a tuple (x, y, z, intensity)
    
    scan_VOI_pts = []
    for pt in scan_points:
        r = sqrt(pt[0]**2 + pt[1]**2)
        z = pt[2]
        if r < LOCAL_MAP_RADIUS and z > LOCAL_MAP_MIN_HEIGHT and z < LOCAL_MAP_MAX_HEIGHT:
            scan_VOI_pts.append([pt[0], pt[1], pt[2]])
    
    scan_VOI_pts = np.array(scan_VOI_pts)
    return scan_VOI_pts

def get_scan_VOI_bin_list(scan_VOI_pts, args, config):
    LOCAL_MAP_RADIUS = config.local_map_radius
    N_RING = config.n_ring
    N_SEG = config.n_seg
 
    # [r, theta], 0<= r <=LOCAL_MAP_RADIUS, 0 <= theta <= 2*pi  
    polar_pts = np.array([[sqrt(pt[0]**2 + pt[1]**2), pi + atan2(pt[1], pt[0])] for pt in scan_VOI_pts])  
    
    r_reso = LOCAL_MAP_RADIUS * 1.0 / N_RING
    angle_reso = 2*pi / N_SEG 
    
    bin_info = [(i*r_reso, (i+1)*r_reso,  j*angle_reso, (j+1)*angle_reso) for i in range(N_RING) for j in range(N_SEG)]
    bin_list = [Bin(*info) for info in bin_info]
    
    for VOI_pt_id, polar_pt in enumerate(polar_pts):
        r, angle = polar_pt[0], polar_pt[1]
        for bin in bin_list:
            if r >= bin.min_r and r < bin.max_r and angle >= bin.min_angle and angle < bin.max_angle:
                bin.point_ids.append(VOI_pt_id)
                break
    return bin_list

def get_basic_bin_list(SOI_pts, args, config):
    LOCAL_MAP_RADIUS = config.local_map_radius
    N_RING = config.n_ring 
    N_SEG = config.n_seg
        
    # [r, theta], 0<= r <=LOCAL_MAP_RADIUS, 0 <= theta <= 2*pi  
    polar_SOI_pts = np.array([[sqrt(pt[0]**2 + pt[1]**2), pi + atan2(pt[1], pt[0])] for pt in SOI_pts])  
    
    r_reso = LOCAL_MAP_RADIUS * 1.0 / N_RING
    angle_reso = 2*pi / N_SEG 
    print('bin resolution, r_reso --> {:.4f} m | angle_reso --> {:.4f} rad'.format(r_reso, angle_reso))
    
    bin_info = [(i*r_reso, (i+1)*r_reso,  j*angle_reso, (j+1)*angle_reso) for i in range(N_RING) for j in range(N_SEG)]
    bin_list = [Bin(*info) for info in bin_info]
    
    for SOI_pt_id, polar_pt in enumerate(polar_SOI_pts):
        r, angle = polar_pt[0], polar_pt[1]
        for bin in bin_list:
            if r >= bin.min_r and r < bin.max_r and angle >= bin.min_angle and angle < bin.max_angle:
                bin.point_ids.append(SOI_pt_id)
                break
    return bin_list

def get_SOI_info(node_msg, pcd_raw_points, pcd_kdtree, args, config):
    LOCAL_MAP_RADIUS = config.local_map_radius
    LOCAL_MAP_MIN_HEIGHT = config.local_map_min_height
    LOCAL_MAP_MAX_HEIGHT = config.local_map_max_height
        
    _, node, _ = node_msg
    # Get local map points
    odom_pose = node.odom
    lidar_pos_in_map = np.zeros((1, 2))
    lidar_pos_in_map[0][0] = odom_pose.position.x
    lidar_pos_in_map[0][1] = odom_pose.position.y

    indices = pcd_kdtree.query_radius(lidar_pos_in_map, r=LOCAL_MAP_RADIUS)
    local_map_pts_raw_ids = indices[0]
    
    local_map_points = np.array([[
        pcd_raw_points[raw_id][0], pcd_raw_points[raw_id][1],
        pcd_raw_points[raw_id][2]
    ] for raw_id in local_map_pts_raw_ids])
    
    if args.dataset == 'kitti':
        transformed_local_map_points = transform_local_map_to_lidar_frame(odom_pose, local_map_points)
    elif args.dataset == 'gazebo':
        transformed_local_map_points = transform_local_map_to_lidar_frame_gazebo(odom_pose, local_map_points)
        
    SOI_pts = []     # coordinates in local map
    SOI_pt_raw_ids = []  # id in local map
    for id, pt in enumerate(transformed_local_map_points):
        z = pt[2]
        if z > LOCAL_MAP_MIN_HEIGHT and z < LOCAL_MAP_MAX_HEIGHT:
            SOI_pts.append(pt)
            SOI_pt_raw_ids.append(local_map_pts_raw_ids[id])
    
    SOI_pts = np.array(SOI_pts)
    return SOI_pts,SOI_pt_raw_ids

def single_bin_scan_ratio_test(bin_pair, scan_VOI_pts, SOI_pts, SOI_pt_raw_ids, args, config):
    scan_bin, submap_bin = bin_pair
    scan_bin_points = np.array([scan_VOI_pts[id] for id in scan_bin.point_ids])
    submap_bin_points = np.array([SOI_pts[id] for id in submap_bin.point_ids])
    
    # Ground plane segmentation
    ground_point_ids, max_plane_z = ransac_bin_ground_segmentaion(submap_bin, SOI_pts, args, config)
    
    submap_bin_static_point_ids = []
    submap_bin_dynamic_point_ids = []
        
    # raw id
    ground_point_raw_ids = [SOI_pt_raw_ids[id] for id in ground_point_ids]
    submap_bin_static_point_raw_ids = [SOI_pt_raw_ids[id] for id in submap_bin_static_point_ids]
    submap_bin_dynamic_point_raw_ids = [SOI_pt_raw_ids[id] for id in submap_bin_dynamic_point_ids]
        
    return ground_point_raw_ids, submap_bin_static_point_raw_ids, submap_bin_dynamic_point_raw_ids

def ransac_bin_ground_segmentaion(bin, SOI_pts, args, config, distance_threshold=0.1, max_iteration=1000):
    """Performs RANSAC ground segmentation

    Args:
        points (np.ndarray): 
        distance_threshold (float, optional): _description_. Defaults to 0.1.
        max_iteration (int, optional): _description_. Defaults to 1000.
    """
    if len(bin.point_ids) < 3:
        return [], np.Inf
    
    bin_points = np.array([SOI_pts[id] for id in bin.point_ids])
    
    mean_z = np.mean(bin_points[:, 2])
    max_z = np.max(bin_points[:, 2])
    min_z = np.min(bin_points[:, 2])
    elevation_height = max_z - min_z
     
    # Estimate the z-threshold
    if args.dataset == 'kitti':    
        if elevation_height < 0.2:
            z_threshold = 0.9*elevation_height + min_z #0.5*mean_z
            distance_threshold *= 2
        else:
            z_threshold = 0.2*elevation_height + min_z
            distance_threshold *= 0.5
    elif args.dataset == 'gazebo':
        if elevation_height < 0.2:
            z_threshold = 0.9*elevation_height + min_z #0.5*mean_z
            distance_threshold *= 2
        else:
            z_threshold = 0.05*elevation_height + min_z
            distance_threshold *= 0.5
                                                
    lower_bin_point_ids = [id for id in bin.point_ids if SOI_pts[id][2] <= z_threshold]
    lower_bin_points = np.array([SOI_pts[id] for id in lower_bin_point_ids]) 
    
    if len(lower_bin_point_ids) < 3:
        return [], np.Inf 
    # print('Starting ransac segmentation, max_z --> {}, min_z --> {}, mean_z --> {}'.format(max_z, min_z, mean_z))
    
    X = lower_bin_points[:, :2]
    z = lower_bin_points[:, 2]
    ransac = RANSACRegressor(base_estimator=None, min_samples=3, residual_threshold=distance_threshold, max_trials=max_iteration)
    ransac.fit(X, z) 
    inlier_mask = ransac.inlier_mask_
    result_ids = np.where(inlier_mask == True)[0]
    
    plane_z_list = [lower_bin_points[id][2] for id in result_ids]
    max_plane_z = max(plane_z_list)
    min_plane_z = min(plane_z_list)
    mean_plane_z = np.mean(plane_z_list)
    
    # Lower than the plane are all treated as ground points
    plane_point_ids = [lower_bin_point_ids[id] for id in result_ids]
    lower_plane_point_ids = [lower_bin_point_ids[id] for id in range(len(lower_bin_points)) if lower_bin_points[id][2] <= min_plane_z]
    
    ground_point_ids = list(set(plane_point_ids) | set(lower_plane_point_ids))
    # Return the indices of the points that belong to the ground
    bin.ground_point_ids = ground_point_ids
    
    return ground_point_ids, max_plane_z 


