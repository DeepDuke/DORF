#! /usr/bin/python2.7

from datetime import datetime

import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from collections import Counter
import random
import numpy as np
from sklearn.decomposition import PCA
from copy import deepcopy

from pypcd import pypcd

import multiprocessing as mp
from functools import partial

from src.utils.color_utils import get_label_color
from src.utils.range_image_utils import sliding_window_get_single_scan_ri, sliding_window_get_residual_ri
from src.utils.range_image_utils import sliding_window_incremental_get_local_map_ri, sliding_window_incremental_get_local_map_info


def coarse_main(pcd_raw_points, pcd_kdtree, pcd_3d_kdtree, node_msg_list, RESULT_SAVING_PATH, args, config):
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    before_revert_dynamic_map_pt_ids = []
    dynamic_map_pt_ids = []
        
    # Multi-Processing
    mp_arg_list = []
    for msg_idx, node_msg in enumerate(node_msg_list):
        mp_arg_list.append((msg_idx, node_msg))
        # sliding_window_incremental_mp_job((msg_idx, node_msg), pcd_raw_points=pcd_raw_points, pcd_kdtree=pcd_kdtree, pcd_3d_kdtree=pcd_3d_kdtree, node_msg_list=node_msg_list, args=args, config=config)
      
    result = mp.Pool(args.n_proc).map(partial(sliding_window_incremental_mp_job, pcd_raw_points=pcd_raw_points, pcd_kdtree=pcd_kdtree, pcd_3d_kdtree=pcd_3d_kdtree, node_msg_list=node_msg_list, args=args, config=config), mp_arg_list) 
   
    print('After multi-processing, result has {} sub-list of dynamic points'.format(len(result)))
    for before_revert_sublist, sublist in result:
        before_revert_dynamic_map_pt_ids.extend(before_revert_sublist)
        dynamic_map_pt_ids.extend(sublist)
        
        before_revert_dynamic_map_pt_ids = list(set(before_revert_dynamic_map_pt_ids))
    
    full_map_pt_ids = list(range(len(pcd_raw_points)))
    
    # Before coarse revert
    before_revert_static_map_pt_ids = []
    before_revert_static_map_pt_ids.extend(list(set(full_map_pt_ids) - set(before_revert_dynamic_map_pt_ids)))
    
    save_dynamic_map(before_revert_dynamic_map_pt_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['coarse_before_revert_dynamic_map_path'])
    print('Saved coarse before revert dynamic map')
    
    save_static_map(before_revert_static_map_pt_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['coarse_before_revert_static_map_path'])
    print('Saved coarse before revert static map')
    
    # After coarse revert
    dynamic_freq_dict = dict(Counter(dynamic_map_pt_ids))   
    
    static_map_pt_ids = []
    dynamic_map_pt_ids = list(set(dynamic_map_pt_ids))
    static_map_pt_ids.extend(list(set(full_map_pt_ids) - set(dynamic_map_pt_ids)))
    
    print('full_map_pt_ids: {} | dynamic_map_pt_ids: {} | staic_map_pt_ids: {}'.
                        format(len(full_map_pt_ids), len(dynamic_map_pt_ids), len(static_map_pt_ids)))
    
    # Save processed map files
    save_dynamic_map(dynamic_map_pt_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['coarse_dynamic_map_path'])
    print('Saved coarse dynamic map')
    
    save_static_map(static_map_pt_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['coarse_static_map_path'])
    print('Saved coarse static map')

    return dynamic_map_pt_ids, dynamic_freq_dict

def sliding_window_incremental_mp_job(iterable_item, pcd_raw_points, pcd_kdtree, pcd_3d_kdtree, node_msg_list, args, config):
    
    msg_idx, node_msg = iterable_item
    topic, cur_node, t = node_msg
    odom_pose = cur_node.odom
    msg_list_length = len(node_msg_list)
    
    SLIDING_WINDOW_SIZE = config.sliding_window_size
    RATIO = config.ratio 
    BEGINNING_RATIO = config.beginning_ratio  
    END_RATIO = config.end_ratio
    
    if msg_idx < SLIDING_WINDOW_SIZE:
        # Larger window at the beginning
        beginning_window = int(max(BEGINNING_RATIO * msg_list_length, RATIO * SLIDING_WINDOW_SIZE))
        beginning_window = min(beginning_window, config.max_window_size)
        batch_msg_ids = list(range(beginning_window))
    elif msg_idx < msg_list_length - SLIDING_WINDOW_SIZE:   
        batch_msg_ids = [msg_idx + delta for delta in list(range(-(SLIDING_WINDOW_SIZE//2), SLIDING_WINDOW_SIZE//2+1, 1)) if (msg_idx + delta) >=0 and (msg_idx + delta) < msg_list_length]
    else:
        # Larger window at the end
        end_window = int(max(END_RATIO * msg_list_length, RATIO * SLIDING_WINDOW_SIZE))
        end_window = min(end_window, config.max_window_size)
        batch_msg_ids = list(range(msg_list_length-1, msg_list_length-end_window-1, -1))
        
    # Shuffle msg ids
    shuffled_batch_msg_ids = deepcopy(batch_msg_ids)
    random.shuffle(shuffled_batch_msg_ids)
    
    # Get local map info
    filtered_local_map_pts_raw_ids, filtered_local_map_pts = sliding_window_incremental_get_local_map_info(msg_idx, odom_pose, pcd_raw_points, pcd_kdtree, args, config)
    
    # Randomly pick a scan for visibility checking
    before_revert_batch_dyn_point_ids = []
    for query_id in shuffled_batch_msg_ids:
        # Generate updated local map range image
        local_map_ri, local_map_index_ri = sliding_window_incremental_get_local_map_ri(
            pcd_raw_points, msg_idx, query_id, filtered_local_map_pts_raw_ids, filtered_local_map_pts, before_revert_batch_dyn_point_ids, args, config)
        
        # Generate single scan range image
        _, query_node, _ = node_msg_list[query_id]
        scan_ri = sliding_window_get_single_scan_ri(msg_idx, cur_node, query_id, query_node, args, config)

        # Get dynamic points from residual image of scan_ri and local_map_ri
        dyn_point_ids = sliding_window_get_residual_ri(msg_idx, query_id, scan_ri, local_map_ri, local_map_index_ri, config)
        before_revert_batch_dyn_point_ids.extend(dyn_point_ids)
        before_revert_batch_dyn_point_ids = list(set(before_revert_batch_dyn_point_ids))
        
    # PCA Coarse Reverting for dynamic points
    batch_dyn_point_ids = coarse_revert(before_revert_batch_dyn_point_ids, pcd_raw_points, pcd_3d_kdtree, config)

    before_revert_local_map_dynamic_point_raw_ids = [filtered_local_map_pts_raw_ids[id] for id in before_revert_batch_dyn_point_ids]
    local_map_dynamic_point_raw_ids = [filtered_local_map_pts_raw_ids[id] for id in batch_dyn_point_ids]
        
    return before_revert_local_map_dynamic_point_raw_ids, local_map_dynamic_point_raw_ids

def coarse_revert(dyn_pt_raw_ids, pcd_raw_points, pcd_3d_kdtree, config):
    """For each dynamic point, first get its static neighbors, compute the static neighbors's biggest eigenvector using PCA,
    than add this dynamic point into its static neighbors, compute their biggest eigenvector using PCA again. Then 
    Args:
        dyn_pt_raw_ids (_type_): coarsely detected possbible dynamic point's id in pcd_raw_map
        pcd_raw_points (_type_): collection of raw pcd global map points 
        pcd_3d_kdtree (_type_): 3D kdtree for raw pcd global map points
    """
    result = map(partial(coarse_revert_single_point, dyn_pt_raw_ids=dyn_pt_raw_ids, pcd_raw_points=pcd_raw_points, pcd_3d_kdtree=pcd_3d_kdtree, config=config), dyn_pt_raw_ids)
    true_dyn_pt_ids = [dyn_pt_raw_ids[i] for i in range(len(dyn_pt_raw_ids)) if result[i] == True]
    
    return true_dyn_pt_ids     

def coarse_revert_single_point(cur_dyn_pt_raw_id, dyn_pt_raw_ids, pcd_raw_points, pcd_3d_kdtree, config):
    # Get all nearest neightbors in a certain radius
    is_true_dynamic = True
    
    pt = pcd_raw_points[cur_dyn_pt_raw_id]
    dyn_pt = np.zeros((1, 3))
    dyn_pt[0][0] = pt[0]
    dyn_pt[0][1] = pt[1]
    dyn_pt[0][2] = pt[2]
    
    indices = pcd_3d_kdtree.query_radius(dyn_pt, r=config.neighbor_radius)
    all_nbr_raw_ids = indices[0]
    static_nbr_raw_ids = list(set(all_nbr_raw_ids) - set(dyn_pt_raw_ids))
    
    if len(static_nbr_raw_ids) < 3:
        return is_true_dynamic 
    
    # Get only static neighbors
    static_nbr_points = np.array([[pcd_raw_points[id][0], pcd_raw_points[id][1], pcd_raw_points[id][2]] for id in static_nbr_raw_ids])
    
    # PCA for only static neighbors
    pca1 = PCA(n_components=3)
    pca1.fit(static_nbr_points)
    eigenvalues1 = pca1.explained_variance_
    
    max_eigenvalue_idx1 = np.argmax(eigenvalues1)
    max_normal1 = pca1.components_[max_eigenvalue_idx1]
    
    # PCA for static neighbors and dynamic point
    union_dyn_nbr_raw_ids = list(set([cur_dyn_pt_raw_id]) | set(static_nbr_raw_ids))
    union_dyn_nbr_points = np.array([[pcd_raw_points[id][0], pcd_raw_points[id][1], pcd_raw_points[id][2]] for id in union_dyn_nbr_raw_ids])
    
    pca2 = PCA(n_components=3)
    pca2.fit(union_dyn_nbr_points)
    eigenvalues2 = pca2.explained_variance_
    
    max_eigenvalue_idx2 = np.argmax(eigenvalues2)
    max_normal2 = pca2.components_[max_eigenvalue_idx2]
    
    # Compare two biggest eigenvectors
    diff = max_normal1 - max_normal2
    dist = np.linalg.norm(diff)
    
    # print('PCA diff --> {}, dist --> {:.10f}'.format(diff, dist))
    
    # If change is smaller than a threshold, then we think it should be a static point, that is we revert this dynamic point to be static
    if dist < config.revert_dist_threshold:
        is_true_dynamic = False
    
    return is_true_dynamic 

def save_static_map(static_map_pt_ids, pcd_raw_points, save_path):
    """Save processed static point cloud map

    Args:
        static_map_pt_ids (list): raw point id for static points
        pcd_raw_points (np.array of tuple, N * 4): each row is a tuple (x, y, z, intensity)
    """
    
    print('There are {} / {} static points reserved'.format(len(static_map_pt_ids), len(pcd_raw_points)))
    
    if len(static_map_pt_ids) == 0:
        print('No static points detected in this step!')
        return 
    
    # Convert static map points to PointCloud2 msg

    points = np.array([[pcd_raw_points[id][0], pcd_raw_points[id][1], pcd_raw_points[id][2],  pcd_raw_points[id][3], get_label_color(np.uint32(pcd_raw_points[id][3]) & 0xFFFF)] 
                        for id in static_map_pt_ids])
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

def save_dynamic_map(dynamic_map_pt_ids, pcd_raw_points, save_path):
    """Save removed dynamic point cloud map

    Args:
        dynamic_map_pt_ids (list): raw point id for dynamic points
        pcd_raw_points (np.array of tuple, N * 4): each row is a tuple (x, y, z, intensity)
    """ 
    
    print('There are {} / {} dynamic points removed'.format(len(dynamic_map_pt_ids), len(pcd_raw_points)))
    
    if len(dynamic_map_pt_ids) == 0:
        print('No dynamic points detected in this step!')
        return 
        
    # Convert static map points to PointCloud2 msg
    points = np.array([[pcd_raw_points[id][0], pcd_raw_points[id][1], pcd_raw_points[id][2],  pcd_raw_points[id][3], get_label_color(np.uint32(pcd_raw_points[id][3]) & 0xFFFF)]
                        for id in dynamic_map_pt_ids])
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


def save_bin_color_map(pcd_raw_points, save_path):
    print('Save bin color raw map')
    # Convert static map points to PointCloud2 msg
    points = np.array([[pt[0], pt[1], pt[2], pt[3], get_label_color(np.uint32(pt[3]) & 0xFFFF)] for pt in pcd_raw_points])
   
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
    