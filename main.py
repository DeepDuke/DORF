#! /usr/bin/python2.7

import os 
import pickle
import argparse
import random
import numpy as np
from datetime import datetime
from tabulate import tabulate
from sklearn.neighbors import KDTree

import rospy
from dorf.utils.pcd_reader import PCDReader
from dorf.utils.rosbag_reader import RosbagReader

from dorf.utils.config_loader import Config

from dorf.utils.my_logger import MyLogger
from dorf.filters.coarse_filter import save_dynamic_map, save_static_map, save_bin_color_map, coarse_filter
from dorf.filters.fine_filter import fine_filter
from dorf.filters.occupancy_checking import occupancy_filter


def log_args():
    global logger, args 
    table = [[key, value] for key, value in vars(args).items()]
    headers = ["parameter name", "value"]
    logger.INFO('Args params are as follows:\n{}'.format(tabulate(table, headers, tablefmt="grid")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='filtering for dynamic points.')
    parser.add_argument('--dataset', type=str, default='kitti', help="dataset type: ['kitti', 'gazebo', 'ust']")
    parser.add_argument('--seq', type=str, default='00', help="Semantic sequence id, ['00', '01', '02', '05', '07', \
                        gazebo sequence id: ['ped_50', 'ped_100', 'ped_150']")
    parser.add_argument('--n_proc', type=int, default=18, help='num of processes')
    parser.add_argument('--seed', type=int, default=0, help='value of random seed')
    parser.add_argument('--coarse_pkl', type=str, default='save', help="save or load  *.pkl for coarse stage ['save', 'load']")
    parser.add_argument('--ground_pkl', type=str, default='save', help="save or load  *.pkl for ground segmentation stage ['save', 'load']")
    parser.add_argument('--config_path', type=str, default='./config/kitti.yaml', help='path of config file')
    
    args = parser.parse_args()
    
    # # ROS init
    rospy.init_node('dorf_node')
    # scan_pub = rospy.Publisher('local_map', PointCloud2, queue_size=10)
    
    config = Config()
    config.load(args.config_path)
    
    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    PCD_FILE_PATH = config.pcd_path['seq_' + args.seq]
    BAG_FILE_PATH = config.bag_path['seq_' + args.seq]
    SEQ_REUSLT_PATH = config.result_path + args.seq + '/'

    if not os.path.exists(SEQ_REUSLT_PATH):
        os.makedirs(SEQ_REUSLT_PATH)
    os.chdir(SEQ_REUSLT_PATH)
    
    # log 
    logger = MyLogger(log_file_path=SEQ_REUSLT_PATH, log_file_name= datetime.now().strftime("%m-%d-%Y_%H-%M-%S") + '_' + config.main_log_name, level=config.log_level)
    log_args()
    logger.INFO('Processing {}-th sequence'.format(args.seq))
    logger.INFO('Current working directory is {}'.format(os.getcwd()))
    
    RESULT_SAVING_PATH = {
        'bin_color_map_path': SEQ_REUSLT_PATH + 'seq_{}_bin_color_map.pcd'.format(args.seq),
        'final_static_map_path': SEQ_REUSLT_PATH + 'seq_{}_final_static_map.pcd'.format(args.seq),
        'final_dynamic_map_path': SEQ_REUSLT_PATH + 'seq_{}_final_dynamic_map.pcd'.format(args.seq),
        'coarse_before_revert_static_map_path': SEQ_REUSLT_PATH + 'seq_{}_coarse_before_revert_static_map.pcd'.format(args.seq),
        'coarse_before_revert_dynamic_map_path': SEQ_REUSLT_PATH + 'seq_{}_coarse_before_revert_dynamic_map.pcd'.format(args.seq),
        'coarse_static_map_path': SEQ_REUSLT_PATH + 'seq_{}_coarse_static_map.pcd'.format(args.seq),
        'coarse_dynamic_map_path': SEQ_REUSLT_PATH + 'seq_{}_coarse_dynamic_map.pcd'.format(args.seq),
        'ground_static_map_path': SEQ_REUSLT_PATH + 'seq_{}_ground_static_map.pcd'.format(args.seq),
        'after_ground_seg_static_map_path': SEQ_REUSLT_PATH + 'seq_{}_after_ground_seg_static_map.pcd'.format(args.seq),
        'after_ground_seg_dynamic_map_path': SEQ_REUSLT_PATH + 'seq_{}_after_ground_seg_dynamic_map.pcd'.format(args.seq),
        'fined_revert_static_map_path': SEQ_REUSLT_PATH + 'seq_{}_fined_revert_static_map.pcd'.format(args.seq),
        'fined_revert_dynamic_map_path': SEQ_REUSLT_PATH + 'seq_{}_fined_revert_dynamic_map.pcd'.format(args.seq),
    }
        
    # Read rosbag
    bag_reader = RosbagReader(BAG_FILE_PATH)
    node_msg_generator = bag_reader.read_bag()
    node_msg_list = list(node_msg_generator)
   
    # Read *.pcd point cloud map file 
    pcd_reader = PCDReader(PCD_FILE_PATH)
    pcd_raw_points = pcd_reader.read_pcd_file().pc_data   # Map points in Map Frame, N*4 points (x, y, z, intensity) label is in intensity
    
    # Only two colors in this pcd, white for static part, red for dynamic part
    save_bin_color_map(pcd_raw_points, RESULT_SAVING_PATH['bin_color_map_path'])
    
    pcd_2d_points = np.array([[point[0], point[1]] for point in pcd_raw_points])
    pcd_kdtree = KDTree(pcd_2d_points)
    
    pcd_3d_points = np.array([[point[0], point[1], point[2]] for point in pcd_raw_points])
    pcd_3d_kdtree = KDTree(pcd_3d_points)
    
    # Store some middle results just for debug
    if args.coarse_pkl == 'save':
        coarse_dynamic_map_pt_ids, coarse_dynamic_freq_dict = coarse_filter(pcd_raw_points, pcd_kdtree, pcd_3d_kdtree, node_msg_list, RESULT_SAVING_PATH, args, config)
        logger.INFO('After coarse removal, there are {} dynamic points'.format(len(set(coarse_dynamic_map_pt_ids))))

        # Store coarse_dynamic_map_pt_ids, coarse_dynamic_freq_dict
        with open(SEQ_REUSLT_PATH + 'seq_{}_coarse_dynamic_map_pt_ids.pkl'.format(args.seq), 'wb') as fp:  # pickling
            pickle.dump(coarse_dynamic_map_pt_ids, fp)
        with open(SEQ_REUSLT_PATH + 'seq_{}_coarse_dynamic_freq_dict.pkl'.format(args.seq), 'wb') as fp:
            pickle.dump(coarse_dynamic_freq_dict, fp)
    else:
        # Load coarse_dynamic_map_pt_ids, coarse_dynamic_freq_dict
        with open(SEQ_REUSLT_PATH + 'seq_{}_coarse_dynamic_map_pt_ids.pkl'.format(args.seq), 'rb') as fp:  # pickling
            coarse_dynamic_map_pt_ids = pickle.load(fp)
        with open(SEQ_REUSLT_PATH + 'seq_{}_coarse_dynamic_freq_dict.pkl'.format(args.seq), 'rb') as fp:
            coarse_dynamic_freq_dict = pickle.load(fp)
    
    if args.ground_pkl == 'save':
        ground_point_raw_ids, ground_freq_dict, fined_static_point_raw_ids, fined_dynamic_point_raw_ids = fine_filter(pcd_raw_points, pcd_kdtree, node_msg_list, RESULT_SAVING_PATH, args, config)
        logger.INFO('After ground segmentation, we get {} ground points'.format(len(ground_point_raw_ids)))
          
        raw_ground_intersection = set(coarse_dynamic_map_pt_ids) & set(ground_point_raw_ids)
        ground_intersection = set([pt_id for pt_id in raw_ground_intersection if coarse_dynamic_freq_dict[pt_id] < ground_freq_dict[pt_id]])
        
        # ground point reverting
        after_seg_dynamic_point_raw_ids = list(set(coarse_dynamic_map_pt_ids) - ground_intersection)
        after_seg_static_point_raw_ids = list(set(range(len(pcd_raw_points))) - set(after_seg_dynamic_point_raw_ids))
        
        logger.INFO('after_seg_dynamic_point_raw_ids={}, after_seg_static_point_raw_ids={}'.format(len(after_seg_dynamic_point_raw_ids), len(after_seg_static_point_raw_ids)))
        logger.INFO('After ground segmentation, we update {}-->{} dynamic points, {}-->{} static points'.format(
            len(coarse_dynamic_map_pt_ids), len(after_seg_dynamic_point_raw_ids), len(pcd_raw_points)-len(coarse_dynamic_map_pt_ids), len(after_seg_static_point_raw_ids)))
        
        save_dynamic_map(after_seg_dynamic_point_raw_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['after_ground_seg_dynamic_map_path'])
        save_static_map(after_seg_static_point_raw_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['after_ground_seg_static_map_path'])
        
        # Store after_seg_dynamic_point_raw_ids, after_seg_dynamic_point_raw_ids
        with open(SEQ_REUSLT_PATH + 'seq_{}_after_seg_dynamic_point_raw_ids.pkl'.format(args.seq), 'wb') as fp:  # pickling
            pickle.dump(after_seg_dynamic_point_raw_ids, fp)
        with open(SEQ_REUSLT_PATH + 'seq_{}_after_seg_static_point_raw_ids.pkl'.format(args.seq), 'wb') as fp:
            pickle.dump(after_seg_static_point_raw_ids, fp)
    else:
        # Load after_seg_dynamic_point_raw_ids, after_seg_dynamic_point_raw_ids
        with open(SEQ_REUSLT_PATH + 'seq_{}_after_seg_dynamic_point_raw_ids.pkl'.format(args.seq), 'rb') as fp:  # pickling
            after_seg_dynamic_point_raw_ids = pickle.load(fp)
        with open(SEQ_REUSLT_PATH + 'seq_{}_after_seg_static_point_raw_ids.pkl'.format(args.seq), 'rb') as fp:
            after_seg_static_point_raw_ids = pickle.load(fp)
    
    # 2D Occupancy Checking
    pure_static_raw_ids = occupancy_filter(pcd_3d_points, node_msg_list, args, config)
    after_occ_checking_dynamic_raw_ids = list(set(after_seg_dynamic_point_raw_ids) - set(pure_static_raw_ids))
    after_occ_checking_static_raw_ids = list(set(range(len(pcd_raw_points))) - set(after_occ_checking_dynamic_raw_ids))
    
    logger.INFO('pure_static_raw_ids = {}'.format(len(pure_static_raw_ids)))
    logger.INFO('after_occ_checking_dynamic_raw_ids = {}, after_occ_checking_static_raw_ids={}'.format(len(after_occ_checking_dynamic_raw_ids), len(after_occ_checking_static_raw_ids)))
    logger.INFO('After 2D Occupancy Checking, we update {}-->{} dynamic points, {}-->{} static points'.format(
        len(after_seg_dynamic_point_raw_ids), len(after_occ_checking_dynamic_raw_ids), len(after_seg_static_point_raw_ids), len(after_occ_checking_static_raw_ids)))
    
    save_dynamic_map(after_occ_checking_dynamic_raw_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['final_dynamic_map_path'])
    save_static_map(after_occ_checking_static_raw_ids, pcd_raw_points, save_path=RESULT_SAVING_PATH['final_static_map_path'])
    