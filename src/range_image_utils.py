#! /usr/bin/python2.7

from color_utils import MOVING_OBJECT_LABELS, COLOR_ZOO, get_random_color, get_label_color, get_moving_object_color, get_static_object_color

from config import remo_params

from tf.transformations import quaternion_matrix, translation_matrix, inverse_matrix
from sensor_msgs import point_cloud2

import numpy as np
from math import sqrt, atan2, asin, pi
from matplotlib import pyplot as plt
from copy import deepcopy


def mark_on_vis_scan_range_ri(raw_points, vis_scan_ri, scan_index_ri):
    """Mark dynamic point on visible scan range image

    Args:
        raw_points (np.ndarray): scan points
        vis_scan_ri (2D np.ndarray): visible scan range image
        scan_index_ri (2D np.ndarray): (row, col) --> idx for point in raw_points

    Returns:
        marked_ri(2D np.ndarray): marked visible scan range image
    """
    marked_ri = deepcopy(vis_scan_ri)
    ele_ids = [(row_id, col_id) for row_id in range(scan_index_ri.shape[0]) for col_id in range(scan_index_ri.shape[1])]
    
    def mark_func(id_tuple):
        row_id, col_id = id_tuple
        tmp_idx = scan_index_ri[row_id][col_id]
        if tmp_idx == -1:
            return
        else:
            point = raw_points[tmp_idx]
            intensity = np.uint32(point[3])

            instance_id = intensity >> 16  # get upper half for instances
            semantic_label = intensity & 0xFFFF
            if semantic_label in MOVING_OBJECT_LABELS:
                marked_ri[row_id, col_id] = np.array([255, 0, 0])
    
    map(mark_func, ele_ids)

    return marked_ri
    
def mark_on_vis_local_map_ri(local_map_pts_raw_ids, pcd_raw_points, vis_local_map_ri, index_ri):
    """Mark dynamic point on visible local map range image

    Args:
        local_map_pts_raw_ids (_type_): _description_
        pcd_raw_points (_type_): _description_
        vis_local_map_ri (_type_): _description_
        index_ri (_type_): _description_

    Returns:
        marked_ri(2D np.ndarray): marked visible local map range image
    """
    marked_ri = deepcopy(vis_local_map_ri)
    ele_ids = [(row_id, col_id) for row_id in range(index_ri.shape[0]) for col_id in range(index_ri.shape[1])]
    
    def mark_func(id_tuple):
        row_id, col_id = id_tuple
        tmp_idx = index_ri[row_id][col_id]
        point_raw_idx = local_map_pts_raw_ids[tmp_idx]

        if point_raw_idx == -1:
            return
        else:
            point = pcd_raw_points[point_raw_idx]
            intensity = np.uint32(point[3])
            instance_id = intensity >> 16  # get upper half for instances
            semantic_label = intensity & 0xFFFF
            if semantic_label in MOVING_OBJECT_LABELS:
                marked_ri[row_id, col_id] = np.array([255, 0, 0])
    
    map(mark_func, ele_ids)
    
    return marked_ri
    
def gen_visible_scan_range_image(scan_range_image):
    """Generate visible scan range image

    Args:
        scan_range_image (2D np.ndarray): scan range image

    Returns:
        vis_img: colorful single scan range image 
    """
    return gen_visible_range_image(scan_range_image)

def gen_visible_local_map_range_image(local_map_range_image):
    """Generate visible local map range image

    Args:
        local_map_range_image (2D np.ndarray): local map range image

    Returns:
        vis_img: colorful local map range image 
    """
    return gen_visible_local_map_range_image(local_map_range_image)
    
def gen_visible_range_image(range_image):
    """Add color on range image for visualization

    Args:
        range_image (2D np.ndarray): raw range image

    Returns:
        vis_img: colorful range image, different color for different depth. Greener is closer while yellower is more far.
    """
    max_r = np.amax(range_image)
    height, width = range_image.shape

    vis_img = np.zeros((height, width, 3), dtype=np.uint8)
    ele_ids = [(row_id, col_id) for row_id in range(height) for col_id in range(width)]
    
    def mark_func(id_tuple):
        row_id, col_id = id_tuple
        r = range_image[row_id, col_id]
        ratio = r / max_r
        color = COLOR_ZOO['deep_green'] + (COLOR_ZOO['yellow'] - COLOR_ZOO['deep_green']) * ratio
        vis_img[row_id, col_id] = np.uint8(color[:])
    
    map(mark_func, ele_ids)

    return vis_img
    
def gen_visible_residual_range_image(scan_ri, local_map_ri, local_map_index_ri, residual_range_image):
        """Generate visible residual range image with colors

        Args:
            scan_ri (2D np.ndarray): scan range image
            local_map_ri (2D np.ndarray): local map range image
            local_map_index_ri (2D np.ndarray): (row, col) --> index in local_map_pts_raw_ids
            residual_range_image (2D np.ndarray): residual of scan range image and local map range image

        Returns:
            res_vis_img(_type_): _description_
            dyn_pt_indices(list): dynamic point id in local_map_pts_raw_ids
        """
        max_r = np.amax(abs(residual_range_image))
        height, width = residual_range_image.shape

        res_vis_img = np.zeros((height, width, 3), dtype=np.uint8)
        dyn_pt_indices = []
        ele_ids = [(row_id, col_id) for row_id in range(height) for col_id in range(width)]

        def mark_func(id_tuple):
            row_id, col_id = id_tuple
            diff = residual_range_image[row_id, col_id]

            if np.isclose(scan_ri[row_id, col_id], 0) or np.isclose(local_map_ri[row_id, col_id], 0):
                # found invalid dynamic point
                res_vis_img[row_id, col_id] = COLOR_ZOO['deep_green']
                return 0
            elif diff > remo_params['DYN_THRESHOLD']:
                # possible dynamic point
                dyn_pt_indices.append(local_map_index_ri[row_id, col_id])
                res_vis_img[row_id, col_id, :] = COLOR_ZOO['red']
                return 1
            elif diff > 0:
                ratio = abs(diff / max_r)
                color = COLOR_ZOO['deep_green'] + (
                    COLOR_ZOO['yellow'] - COLOR_ZOO['deep_green']) * ratio
                res_vis_img[row_id, col_id, :] = np.uint8(color[:])
                return 0
            else:
                # diff < 0
                ratio = abs(diff / max_r)
                color = COLOR_ZOO['deep_green'] + (
                    COLOR_ZOO['yellow'] - COLOR_ZOO['deep_green']) * ratio
                res_vis_img[row_id, col_id, :] = np.uint8(color[:])
                return 0
            
        dyn_cnt  = sum(map(mark_func, ele_ids))

        print('gen_visible_residual_range_image found {} dynamic points'.format(dyn_cnt))

        return res_vis_img, dyn_pt_indices
    
def gen_local_map_range_image(msg_idx, odom_pose, pcd_raw_points, pcd_kdtree, args):
    """Generate range image local map 

    Args:
        msg_idx (int): node msg idx
        odom_pose (geometry_msgs/Pose): robot body pose in map frame
        pcd_raw_points (np.ndarray): N*4 (x, y, z, intensity)
        pcd_kdtree (sklearn.neighbors,KDTree): kdtree for find local map points in a certain radius on XY plane

    Returns:
        local_map_ri(2D np.ndarray): local map range image
        local_map_pts_raw_ids(iterable): raw index for point in local map
        index_ri: (row, col) --> index in local_map_pts_raw_ids
    """
    # Find local map points in a certain range
    lidar_pos_in_map = np.zeros((1, 2))
    lidar_pos_in_map[0][0] = odom_pose.position.x
    lidar_pos_in_map[0][1] = odom_pose.position.y
    # lidar_pos_in_map[0][2] = odom_pose.position.z

    if args.dataset == 'kitti':
        LOCAL_MAP_RADIUS = remo_params['LOCAL_MAP_RADIUS']
    elif args.dataset == 'gazebo':
        LOCAL_MAP_RADIUS = remo_params['gazebo_local_map_radius']
    elif args.dataset == 'ust':
        LOCAL_MAP_RADIUS = remo_params['ust_local_map_radius']
        
    indices = pcd_kdtree.query_radius(lidar_pos_in_map, r=LOCAL_MAP_RADIUS)
    local_map_pts_raw_ids = indices[0]
    local_map_points = np.array([[
        pcd_raw_points[raw_id][0], pcd_raw_points[raw_id][1],
        pcd_raw_points[raw_id][2]
    ] for raw_id in local_map_pts_raw_ids])
    # np.savetxt('{}_local_map_points.csv'.format(msg_idx), local_map_points, fmt='%.2f', delimiter=',')
    
    # Transform pcd map points to lidar frame
    if args.dataset == 'kitti':
        transformed_local_map_points = transform_local_map_to_lidar_frame(odom_pose, local_map_points)
    elif args.dataset == 'gazebo' or args.dataset == 'ust':
        transformed_local_map_points = transform_local_map_to_lidar_frame_gazebo(odom_pose, local_map_points)
        
    # np.savetxt('{}_trans_local_map_points.csv'.format(msg_idx), transformed_local_map_points, fmt='%.2f', delimiter=',')
    # self._logger.DEBUG('z_min --> {} | z_max --> {}'.format(np.amin(transformed_local_map_points[:, 2]), np.amax(transformed_local_map_points[:, 2])))

    # Filter transformed local map points by height
    
    if args.dataset == 'kitti':
        LOCAL_MAP_MIN_HEIGHT = remo_params['LOCAL_MAP_MIN_HEIGHT']
        LOCAL_MAP_MAX_HEIGHT = remo_params['LOCAL_MAP_MAX_HEIGHT']
    elif args.dataset == 'gazebo':
        LOCAL_MAP_MIN_HEIGHT = remo_params['gazebo_local_map_min_height'] 
        LOCAL_MAP_MAX_HEIGHT = remo_params['gazebo_local_map_max_height'] 
    elif args.dataset == 'ust':
        LOCAL_MAP_MIN_HEIGHT = remo_params['ust_local_map_min_height']
        LOCAL_MAP_MAX_HEIGHT = remo_params['ust_local_map_max_height']
    
    filtered_transformed_local_map_pts = []
    filtered_local_map_pts_raw_ids = []
    for idx, pt in enumerate(transformed_local_map_points):
        if pt[2] > LOCAL_MAP_MIN_HEIGHT and pt[2] < LOCAL_MAP_MAX_HEIGHT:
            filtered_transformed_local_map_pts.append(pt)
            filtered_local_map_pts_raw_ids.append(local_map_pts_raw_ids[idx])
        else:
            # self._logger.DEBUG('filtered local map points --> {}'.format(pt))
            pass 
    filtered_transformed_local_map_pts = np.array(filtered_transformed_local_map_pts)
    print('{}-frame. After filtering by the height, we have {}/{} points in local map'.format(msg_idx, len(filtered_transformed_local_map_pts), len(transformed_local_map_points)))
    
    # Generate range image
    local_map_ri, index_ri = gen_range_image(filtered_transformed_local_map_pts, args)

    return local_map_ri, filtered_local_map_pts_raw_ids, index_ri
    
def transform_local_map_to_lidar_frame(odom_pose, local_map_points):
    """Transform local map points into LiDAR Frame

    Args:
        odom_pose (geometry_msgs/Pose): body pose in map frame
        local_map_points (2D np.ndarray): N*3 (x, y, z)

    Returns:
        (2D np.ndarray) N*3 (x, y, z): transformed local_map points in LiDAR Frame
    """
    ## Body frame to Map frame
    # Homogeneous rotation matrix
    Mr_body2map = quaternion_matrix([
        odom_pose.orientation.x, odom_pose.orientation.y,
        odom_pose.orientation.z, odom_pose.orientation.w
    ])

    # Homogeneous translation matrix
    Mt_body2map = translation_matrix(
        np.array([
            odom_pose.position.x, odom_pose.position.y,
            odom_pose.position.z
        ]))

    # Homogeneous transformation matrix
    T_body2map = np.zeros((4, 4))
    T_body2map[0:3, 0:3] = Mr_body2map[0:3, 0:3]
    T_body2map[:, 3] = Mt_body2map[:, 3]
    T_map2body = inverse_matrix(T_body2map)

    ## LiDAR Frame to Body Frame
    # Homogeneous rotation matrix
    Mr_lidar2body = quaternion_matrix([0.0, 0.0, 0.0, 1.0])

    # Homogeneous translation matrix
    Mt_lidar2body = translation_matrix(np.array([0.0, 0.0, 1.73]))

    # Homogeneous transformation matrix
    T_lidar2body = np.zeros((4, 4))
    T_lidar2body[0:3, 0:3] = Mr_lidar2body[0:3, 0:3]
    T_lidar2body[:, 3] = Mt_lidar2body[:, 3]
    T_body2lidar = inverse_matrix(T_lidar2body)

    # Map Frame to LiDAR Frame
    T_map2lidar = T_body2lidar.dot(T_map2body)
    # Convert local_map_points to Homogeneous Coordinates
    homo_local_map_points = np.array([[p[0], p[1], p[2], 1]
                                        for p in local_map_points])
    # Transform to lidar frame
    transformed_homo_points = (T_map2lidar.dot(homo_local_map_points.T)).T

    return transformed_homo_points[:, :3]

def transform_local_map_to_lidar_frame_gazebo(lidar_pose, local_map_points):
    """Transform local map points into LiDAR Frame

    Args:
        lidar_pose (geometry_msgs/Pose): lidar pose in map frame
        local_map_points (2D np.ndarray): N*3 (x, y, z)

    Returns:
        (2D np.ndarray) N*3 (x, y, z): transformed local_map points in LiDAR Frame
    """
    ## Body frame to Map frame
    # Homogeneous rotation matrix
    Mr_lidar2map = quaternion_matrix([
        lidar_pose.orientation.x, lidar_pose.orientation.y,
        lidar_pose.orientation.z, lidar_pose.orientation.w
    ])

    # Homogeneous translation matrix
    Mt_lidar2map = translation_matrix(
        np.array([
            lidar_pose.position.x, lidar_pose.position.y,
            lidar_pose.position.z
        ]))

    # Homogeneous transformation matrix
    T_lidar2map = np.zeros((4, 4))
    T_lidar2map[0:3, 0:3] = Mr_lidar2map[0:3, 0:3]
    T_lidar2map[:, 3] = Mt_lidar2map[:, 3]
    T_map2lidar = inverse_matrix(T_lidar2map)

    # Convert local_map_points to Homogeneous Coordinates
    homo_local_map_points = np.array([[p[0], p[1], p[2], 1]
                                        for p in local_map_points])
    # Transform to lidar frame
    transformed_homo_points = (T_map2lidar.dot(homo_local_map_points.T)).T

    return transformed_homo_points[:, :3]

def gen_scan_range_image(points, args):
    """Convert points from PointCloud2 into numpy range image

    Args:
        points (list): list of tuple, each tuple is like Point(x=22.930999755859375, y=0.06700112670660019, z=0.9849987030029297, intensity=70.0)

    Returns:
        np.array: range_image
    """
    return gen_range_image(points, args)
    
def gen_range_image(points, args):
    """Convert numpy array points into range image
    Semantic Kitti Dataset uses Velodyne HDL-64E LiDAR:
                Horizontal          Vertical
    FOV:        360 degrees         26.8 (+2 ~ -24.8) degrees
    resolution: 0.08 degree         0.04 degree
    Args:
        points (list): list of tuple, each tuple is like Point(x=22.930999755859375, y=0.06700112670660019, z=0.9849987030029297, intensity=70.0)

    Returns:
        range image(2D np.ndarray): range image
        index_ri(2D np.ndarray): (row, col) --> point id in points 
    """    
    if args.dataset == 'kitti':
        RANGE_IMAGE_SHAPE = remo_params['RANGE_IMAGE_SHAPE']
    elif args.dataset == 'gazebo':
        RANGE_IMAGE_SHAPE = remo_params['gazebo_range_image_shape']
    elif args.dataset == 'ust':
        RANGE_IMAGE_SHAPE = remo_params['ust_range_image_shape']
        
    range_image = np.full(RANGE_IMAGE_SHAPE, 0.0)

    max_phi = -100
    min_phi = 100
    max_theta = -100
    min_theta = 100
    max_r = -100
    # point index in points that finally contributes to range image
    index_ri = np.full(RANGE_IMAGE_SHAPE, -1)

    # calc range image pixel coordinates (u, v) for each point (x, y, z)
    for idx, point in enumerate(points):
        row_id, col_id, phi, theta, r = calc_range_image_coordinates(point, args)
        # save closest point
        if np.isclose(range_image[row_id, col_id], 0.0):
            range_image[row_id, col_id] = r
            index_ri[row_id, col_id] = idx
        else:
            if r < range_image[row_id, col_id]:
                range_image[row_id, col_id] = r
                index_ri[row_id, col_id] = idx

        max_phi = max(max_phi, phi)
        min_phi = min(min_phi, phi)
        max_theta = max(max_theta, theta)
        min_theta = min(min_theta, theta)
        max_r = max(max_r, r)

    print(
        'max_r --> {:.4f} | max_phi --> {:.4f} | min_phi --> {:.4f} | max_theta --> {:.4f} | min_theta --> {:.4f}'
        .format(max_r, max_phi, min_phi, max_theta, min_theta))

    return range_image, index_ri
    
def calc_range_image_coordinates(point, args):
    """Calculate range image (row_id, col_id) for given 3D point

    Args:
        point (iterable): first three elements should be coordinates (x, y, z, ...)

    Returns:
        tuple: (row_id, col_id, phi, theta, r)
    """
    if args.dataset == 'kitti':
        HORIZONTAL_FOV = remo_params['HORIZONTAL_FOV']
        THETA_UP = remo_params['THETA_UP']
        VERTICAL_FOV = remo_params['VERTICAL_FOV']
        RANGE_IMAGE_WIDTH = remo_params['RANGE_IMAGE_WIDTH']
        RANGE_IMAGE_HEIGHT = remo_params['RANGE_IMAGE_HEIGHT']
    elif args.dataset == 'gazebo':
        HORIZONTAL_FOV = remo_params['gazebo_horizontal_fov']
        THETA_UP = remo_params['gazebo_theta_up']
        VERTICAL_FOV = remo_params['gazebo_vertical_fov']
        RANGE_IMAGE_WIDTH = remo_params['gazebo_range_image_width']
        RANGE_IMAGE_HEIGHT = remo_params['gazebo_range_image_height']
    elif args.dataset == 'ust':
        HORIZONTAL_FOV = remo_params['ust_horizontal_fov']
        THETA_UP = remo_params['ust_theta_up']
        VERTICAL_FOV = remo_params['ust_vertical_fov']
        RANGE_IMAGE_WIDTH = remo_params['ust_range_image_width']
        RANGE_IMAGE_HEIGHT = remo_params['ust_range_image_height']

    x, y, z = point[0], point[1], point[2]
    r = sqrt(x**2 + y**2 + z**2)

    phi = atan2(y, x)  # azmuth angle [-pi, pi]
    theta = asin(z / r)  # elevation angle (-pi/2, pi/2)

    # u = int((pi + phi) / horizontal_FOV * width) - 1  # column
    col_id = int((1 - phi / (HORIZONTAL_FOV / 2)) * RANGE_IMAGE_WIDTH / 2)  # column
    row_id = int((THETA_UP - theta) / VERTICAL_FOV * RANGE_IMAGE_HEIGHT)    # row

    # Add bound limit
    col_id = max(0, min(col_id, RANGE_IMAGE_WIDTH - 1))
    row_id = max(0, min(row_id, RANGE_IMAGE_HEIGHT - 1))

    return row_id, col_id, phi, theta, r

############################################################  Sliding Window Functions ############################################################

def sliding_window_get_single_scan_ri(msg_idx, cur_node, query_id, query_node, args):
    """Get single scan related range images
    Args:
        msg_idx (int): node msg index
        node (ERASOR node msg): contains odom, scan ...
    Returns:
        scan_ri(2D np.ndarray): scan range image
    """
    # Generate scan range image
    pc_msg = query_node.lidar
    points = point_cloud2.read_points_list(pc_msg)

    # Transform points in query_node LiDAR Frame to cur_node LiDAR Frame
    query_odom_pose = query_node.odom
    cur_odom_pose = cur_node.odom
    transfomed_points = transform_from_one_lidar_frame_to_another_lidar_frame(query_odom_pose, cur_odom_pose, points, args)

    print('{}-frame, query-di --> {}, Generate scan range image ...'.format(msg_idx, query_id))
    
    scan_ri, scan_index_ri = gen_scan_range_image(transfomed_points, args)
    # np.savetxt('./{}_{}_scan_ri.csv'.format(msg_idx, query_id), scan_ri, fmt='%.2f', delimiter=',')

    # # Generate visible coloful range image
    # vis_scan_ri = gen_visible_range_image(scan_ri)
    # plt.imsave('./{}_{}_scan_ri.jpg'.format(msg_idx, query_id), vis_scan_ri)

    # # Mark dynamic points on vis_scan_ri, here we need to get intensity from points
    # marked_vis_scan_ri = mark_on_vis_scan_range_ri(points, vis_scan_ri, scan_index_ri)
    # plt.imsave('./{}_{}_marked_vis_scan_ri.jpg'.format(msg_idx, query_id), marked_vis_scan_ri)
    
    return scan_ri


def sliding_window_get_residual_ri(msg_idx, query_id, scan_ri, local_map_ri, local_map_index_ri):
    """Generate residual range image using scan range image and local_map range image 
    Args:
        msg_idx (int): idx for node msg
        scan_ri (2D np.ndarray): scan range image
        local_map_ri (2D np.ndarray): local_map range map 
        local_map_index_ri (2D np.ndarray): recording of each point's raw index in local_map_pts_raw_ids
    Returns:
        list: list of idx for dynamic map points in local_map
    """
    print('{}-frame, query_id --> {}, Generate residual range image ...'.format(msg_idx, query_id))
    residual_ri = scan_ri - local_map_ri
    vis_residual_ri, dyn_point_ids = gen_visible_residual_range_image(scan_ri, local_map_ri, local_map_index_ri, residual_ri)
    # plt.imsave('./{}_{}_vis_residual_ri.jpg'.format(msg_idx, query_id), vis_residual_ri)
    print('dyn_point_ids length --> {}'.format(len(dyn_point_ids)))
    
    return dyn_point_ids

def transform_from_one_lidar_frame_to_another_lidar_frame(query_odom_pose, cur_odom_pose, points, args):
    if args.dataset == 'kitti':
        T_query_lidar2map = get_lidar_to_map_transformation_matrix(query_odom_pose)
        T_cur_lidar2map = get_lidar_to_map_transformation_matrix(cur_odom_pose)
    elif args.dataset == 'gazebo' or args.dataset == 'ust':
        T_query_lidar2map = get_lidar_to_map_transformation_matrix_gazebo(query_odom_pose)
        T_cur_lidar2map = get_lidar_to_map_transformation_matrix_gazebo(cur_odom_pose)
    
    T_map2cur_lidar = inverse_matrix(T_cur_lidar2map)
    
    T_query_lidar2cur_lidar = T_map2cur_lidar.dot(T_query_lidar2map)
    
    homogeneous_points = np.array([[pt[0], pt[1], pt[2], 1] for pt in points])
    transformed_points = (T_query_lidar2cur_lidar.dot(homogeneous_points.T)).T
    
    return transformed_points[:, :3]

def transform_from_lidar_frame_to_map_frame(odom_pose, scan_points, args):
    if args.dataset == 'kitti':
        T_lidar2map = get_lidar_to_map_transformation_matrix(odom_pose)
    elif args.dataset == 'gazebo' or args.dataset == 'ust':
        T_lidar2map = get_lidar_to_map_transformation_matrix_gazebo(odom_pose)
        
    homogeneous_points = np.array([[pt[0], pt[1], pt[2], 1] for pt in scan_points])
    transformed_points = (T_lidar2map.dot(homogeneous_points.T)).T
    
    return transformed_points[:, :3] 

def get_lidar_to_map_transformation_matrix(odom_pose):
    ## Body Frame to Map Frame
    # Homogeneous rotation matrix
    Mr_body2map = quaternion_matrix([
        odom_pose.orientation.x, odom_pose.orientation.y,
        odom_pose.orientation.z, odom_pose.orientation.w
    ])
    
    # Homogeneous translation matrix
    Mt_body2map = translation_matrix(
        np.array([
            odom_pose.position.x, odom_pose.position.y,
            odom_pose.position.z
        ]))

    # Homogeneous transformation matrix
    T_body2map = np.zeros((4, 4))
    T_body2map[0:3, 0:3] = Mr_body2map[0:3, 0:3]
    T_body2map[:, 3] = Mt_body2map[:, 3]
    
    ## LiDAR Frame to Body Frame
    # Homogeneous rotation matrix
    Mr_lidar2body = quaternion_matrix([0.0, 0.0, 0.0, 1.0])

    # Homogeneous translation matrix
    Mt_lidar2body = translation_matrix(np.array([0.0, 0.0, 1.73]))

    # Homogeneous transformation matrix
    T_lidar2body = np.zeros((4, 4))
    T_lidar2body[0:3, 0:3] = Mr_lidar2body[0:3, 0:3]
    T_lidar2body[:, 3] = Mt_lidar2body[:, 3]
    
    T_lidar2map = T_body2map.dot(T_lidar2body)
    
    return T_lidar2map

def get_lidar_to_map_transformation_matrix_gazebo(odom_pose):
    ## Lidar Frame to Map Frame, for Gazebo Dataset, odom_pose = robot_pose = lidar_pose in global map frame
    # Homogeneous rotation matrix
    Mr_lidar2map = quaternion_matrix([
        odom_pose.orientation.x, odom_pose.orientation.y,
        odom_pose.orientation.z, odom_pose.orientation.w
    ])
    
    # Homogeneous translation matrix
    Mt_lidar2map = translation_matrix(
        np.array([
            odom_pose.position.x, odom_pose.position.y,
            odom_pose.position.z
        ]))

    # Homogeneous transformation matrix
    T_lidar2map = np.zeros((4, 4))
    T_lidar2map[0:3, 0:3] = Mr_lidar2map[0:3, 0:3]
    T_lidar2map[:, 3] = Mt_lidar2map[:, 3]
    
    return T_lidar2map

############################################################  Sliding Window Incremental Functions ############################################################

def sliding_window_incremental_get_local_map_ri(pcd_raw_points, msg_idx, query_id, 
                                                filtered_local_map_pts_raw_ids, filtered_local_map_pts, batch_dyn_point_ids, args):
    """raw whole map points in *.pcd file

    Args:
        pcd_raw_points (2D np.ndarray):N*4, (x, y, z, intensity)
        pcd_kdtree (sklearn KDTree): used for get local map points in a certain radisu on xy plane
        msg_idx (int): node msg idx
        odom_pose (geometry_msgs/Pose): body pose in map frame

    Returns:
        local_map_ri(2D np.ndarray): local map range image
        local_map_pts_raw_ids(list): list of ids for local map points in raw whole map 
        local_map_index_ri(2D np.ndarray): (row, col) --> point id in local_map_pts_raw_ids
        vis_local_map_ri(2D np.ndarray): visbile colorful local map range image
    """
    print('{}-th frame, query_id --> {}, Generate local map range image ...'.format(msg_idx, query_id))
    
    local_map_ri, local_map_index_ri = sliding_window_incremental_gen_local_map_range_image(filtered_local_map_pts, batch_dyn_point_ids, args)
    # np.savetxt('{}_local_map_ri.csv'.format(msg_idx), local_map_ri, fmt='%.2f', delimiter=',')

    # # Generate visible coloful range image
    # vis_local_map_ri = gen_visible_range_image(local_map_ri)
    # plt.imsave('./{}_{}_local_map_ri.jpg'.format(msg_idx, query_id), vis_local_map_ri)

    # # Mark dynamic points on vis_local_map_ri
    # marked_vis_local_map_ri = mark_on_vis_local_map_ri(filtered_local_map_pts_raw_ids, pcd_raw_points, vis_local_map_ri, local_map_index_ri)
    # plt.imsave('./{}_{}_marked_vis_local_map_ri.jpg'.format(msg_idx, query_id), marked_vis_local_map_ri)
    
    return local_map_ri, local_map_index_ri

def sliding_window_incremental_gen_local_map_range_image(filtered_local_map_pts, batch_dyn_point_ids, args):
    """Generate range image local map 

    Args:
        msg_idx (int): node msg idx
        odom_pose (geometry_msgs/Pose): robot body pose in map frame
        pcd_raw_points (np.ndarray): N*4 (x, y, z, intensity)
        pcd_kdtree (sklearn.neighbors,KDTree): kdtree for find local map points in a certain radius on XY plane

    Returns:
        local_map_ri(2D np.ndarray): local map range image
        local_map_pts_raw_ids(iterable): raw index for point in local map
        index_ri: (row, col) --> index in local_map_pts_raw_ids
    """ 
    # Excluding previous detected dynamic points
    
    updated_local_map_pt_ids = list(set(range(len(filtered_local_map_pts))) - set(batch_dyn_point_ids))
    updated_local_map_pts = [filtered_local_map_pts[id] for id in updated_local_map_pt_ids]
    
    # Generate range image    
    if args.dataset == 'kitti':
        RANGE_IMAGE_SHAPE = remo_params['RANGE_IMAGE_SHAPE'] 
    elif args.dataset == 'gazebo':
        RANGE_IMAGE_SHAPE = remo_params['gazebo_range_image_shape'] 
    elif args.dataset == 'ust':
        RANGE_IMAGE_SHAPE = remo_params['ust_range_image_shape']

    local_map_ri = np.full(RANGE_IMAGE_SHAPE, 0.0)
    local_map_index_ri = np.full(RANGE_IMAGE_SHAPE, -1)  # point index in points that finally contributes to range image

    # calc range image pixel coordinates (u, v) for each point (x, y, z)
    for idx, point in zip(updated_local_map_pt_ids, updated_local_map_pts):
        row_id, col_id, phi, theta, r = calc_range_image_coordinates(point, args)
        # save closest point
        if np.isclose(local_map_ri[row_id, col_id], 0.0):
            local_map_ri[row_id, col_id] = r
            local_map_index_ri[row_id, col_id] = idx
        else:
            if r < local_map_ri[row_id, col_id]:
                local_map_ri[row_id, col_id] = r
                local_map_index_ri[row_id, col_id] = idx

    return local_map_ri, local_map_index_ri

def sliding_window_incremental_get_local_map_info(msg_idx, odom_pose, pcd_raw_points, pcd_kdtree, args):
    # Find local map points in a certain range
    lidar_pos_in_map = np.zeros((1, 2))
    lidar_pos_in_map[0][0] = odom_pose.position.x
    lidar_pos_in_map[0][1] = odom_pose.position.y
    # lidar_pos_in_map[0][2] = odom_pose.position.z

    if args.dataset == 'kitti':
        LOCAL_MAP_RADIUS = remo_params['LOCAL_MAP_RADIUS']
        LOCAL_MAP_MIN_HEIGHT = remo_params['LOCAL_MAP_MIN_HEIGHT']
        LOCAL_MAP_MAX_HEIGHT = remo_params['LOCAL_MAP_MAX_HEIGHT']
    elif args.dataset == 'gazebo':
        LOCAL_MAP_RADIUS = remo_params['gazebo_local_map_radius']
        LOCAL_MAP_MIN_HEIGHT = remo_params['gazebo_local_map_min_height'] 
        LOCAL_MAP_MAX_HEIGHT = remo_params['gazebo_local_map_max_height'] 
    elif args.dataset == 'ust':
        LOCAL_MAP_RADIUS = remo_params['ust_local_map_radius']
        LOCAL_MAP_MIN_HEIGHT = remo_params['ust_local_map_min_height'] 
        LOCAL_MAP_MAX_HEIGHT = remo_params['ust_local_map_max_height'] 
        
    indices = pcd_kdtree.query_radius(lidar_pos_in_map, r=LOCAL_MAP_RADIUS)
    local_map_pts_raw_ids = indices[0]
    local_map_points = np.array([[
        pcd_raw_points[raw_id][0], pcd_raw_points[raw_id][1],
        pcd_raw_points[raw_id][2]
    ] for raw_id in local_map_pts_raw_ids])
    # np.savetxt('{}_local_map_points.csv'.format(msg_idx), local_map_points, fmt='%.2f', delimiter=',')
    
    # Transform pcd map points to lidar frame
    if args.dataset == 'kitti':
        transformed_local_map_points = transform_local_map_to_lidar_frame(odom_pose, local_map_points)
    elif args.dataset == 'gazebo' or args.dataset == 'ust':
        transformed_local_map_points = transform_local_map_to_lidar_frame_gazebo(odom_pose, local_map_points)
        
    # np.savetxt('{}_trans_local_map_points.csv'.format(msg_idx), transformed_local_map_points, fmt='%.2f', delimiter=',')

    # Filter transformed local map points by height
    filtered_local_map_pts_raw_ids = []
    filtered_local_map_pts = []
    for idx, pt in enumerate(transformed_local_map_points):
        # Height Filtering
        if pt[2] > LOCAL_MAP_MIN_HEIGHT and pt[2] < LOCAL_MAP_MAX_HEIGHT:
            filtered_local_map_pts_raw_ids.append(local_map_pts_raw_ids[idx])     
            filtered_local_map_pts.append(pt)
    print('{}-frame. After filtering by the height, we have {}/{} points in local map'.format(msg_idx, len(filtered_local_map_pts), len(transformed_local_map_points)))

    return filtered_local_map_pts_raw_ids, filtered_local_map_pts