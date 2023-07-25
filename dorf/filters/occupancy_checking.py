#! /usr/bin/python2.7

import math 
import numpy as np 
from bresenham import bresenham
from functools import partial
import multiprocessing as mp

from sensor_msgs import point_cloud2
from dorf.utils.range_image_utils import transform_from_lidar_frame_to_map_frame


class Grid:
    def __init__(self):
        self.point_ids = []
        self.boundary = [0, 0, 0, 0]  # (x_min, y_min, x_max, y_max)
        self.occ_cnt = 0
        self.pass_cnt = 0
        self.prob = 0
        
    
class OccupancyGridMap2D:
    def __init__(self, pcd_3d_points, reso, args):
        self.resolution = reso 
        self.args = args
        self._init_2D_grid_map(pcd_3d_points)
        
    def _init_2D_grid_map(self, pcd_3d_points):
        # Map Boundary
        self.x_min = np.amin(pcd_3d_points[:, 0])
        self.x_max = np.amax(pcd_3d_points[:, 0])
        self.y_min = np.amin(pcd_3d_points[:, 1])
        self.y_max = np.amax(pcd_3d_points[:, 1])
        
        print('Boundary ({}, {}, {}, {})'.format(self.x_min, self.y_min, self.x_max, self.y_max))
        
        # Grid Generation
        self.X_GRID_NUM = int(math.ceil((self.x_max - self.x_min) / self.resolution * 1.0))
        self.Y_GRID_NUM = int(math.ceil((self.y_max - self.y_min) / self.resolution * 1.0))
        self.grid_map = [[self._init_grid(u, v) for v in range(self.Y_GRID_NUM)] for u in range(self.X_GRID_NUM)]
        
        # Add Point Info To Each Grid
        for id, point in enumerate(pcd_3d_points):
            u, v = self.get_grid_coord(point)
            grid = self.grid_map[u][v]
            grid.point_ids.append(id)
    
    def _init_grid(self, u, v):
        # Calculate Real Boudary For 3D Point
        grid_x_min = self.x_min + u * self.resolution
        grid_x_max = self.x_min + (u + 1) * self.resolution
        grid_y_min = self.y_min + v * self.resolution
        grid_y_max = self.y_min + (v + 1) * self.resolution
        
        grid = Grid()
        grid.boundary = [grid_x_min, grid_x_max, grid_y_min, grid_y_max]

        return grid 
    
    def get_grid_coord(self, point):
        # Get 2D Grid Coordinate for 3D Point
        x, y = point[0], point[1]
        u = int(math.ceil((x - self.x_min) / (self.resolution * 1.0)))
        v = int(math.ceil((y - self.y_min) / (self.resolution * 1.0)))
        
        u = min(u, self.X_GRID_NUM-1)
        v = min(v, self.Y_GRID_NUM-1)
        # print('point --> {}, (u, v) --> ({}, {}), X_GRID_NUM --> {}, Y_GRID_NUM --> {}'.format(
        #     point, u, v, self.X_GRID_NUM, self.Y_GRID_NUM))
        
        return u, v
    
    def get_grid(self, u, v):
        return self.grid_map[u][v]
    
    def get_point_ids_in_a_grid(self, u, v):
        grid = self.grid_map[u][v]
        return grid.point_ids
    
    def get_ray_tracing_result(self, cur_pose, scan_points):
        # TODO: transformed scan_points to map frame
        transformed_scan_points = transform_from_lidar_frame_to_map_frame(cur_pose, scan_points, self.args)
        
        single_scan_middle_grid_coords = []
        for point in transformed_scan_points:
            # Ray Start Grid
            start_u, start_v = self.get_grid_coord((cur_pose.position.x, cur_pose.position.y))
            start_grid = self.grid_map[start_u][start_v]
            # Ray End Grid
            end_u, end_v = self.get_grid_coord(point)
            end_grid = self.grid_map[end_u][end_v]
            
            # Bresenham Algorithm To Draw Middle Grids
            middle_grid_coords = list(bresenham(start_u, start_v, end_u, end_v))[:-1]  # remove end grid

            single_scan_middle_grid_coords.extend(middle_grid_coords)
            
        return single_scan_middle_grid_coords

def single_scan_ray_tracing(node_msg, occupancy_grid_map):
    ray_tracing_result = []
    
    _, msg, _ = node_msg
    cur_odom = msg.odom
    pc2_msg = msg.lidar
    scan_points = point_cloud2.read_points(pc2_msg)  # (x, y, z)
    single_scan_middle_grid_coords = occupancy_grid_map.get_ray_tracing_result(cur_odom, scan_points)
    ray_tracing_result.extend(single_scan_middle_grid_coords)
    
    return ray_tracing_result 

def occupancy_filter(pcd_3d_points, node_msg_list, args, config):
    occupancy_grid_map = OccupancyGridMap2D(pcd_3d_points, config.ray_tracing_resolution, args)
    result = mp.Pool(args.n_proc).map(partial(single_scan_ray_tracing, occupancy_grid_map=occupancy_grid_map), node_msg_list)
    
    ray_tracing_result = []
    for res in result:
        ray_tracing_result.extend(res)
    
     # Calculate The Occupancy Probability
    for u, v in ray_tracing_result:
        grid = occupancy_grid_map.get_grid(u, v)
        grid.pass_cnt += 1
    
    pure_static_raw_ids = []
    PURE_STATIC_THRESHOLD = config.pure_static_threshold
        
    for u in range(occupancy_grid_map.X_GRID_NUM):
        for v in range(occupancy_grid_map.Y_GRID_NUM):
            grid = occupancy_grid_map.get_grid(u, v)
            grid.occ_cnt = len(grid.point_ids)
            grid.prob = grid.occ_cnt / (grid.occ_cnt + grid.pass_cnt + 1e-6)
            print('Grid ({}, {}), occ_cnt = {},  pass_cnt = {}'.format(u, v, grid.occ_cnt, grid.pass_cnt))
            if grid.prob > PURE_STATIC_THRESHOLD:
                pure_static_raw_ids.extend(grid.point_ids)
    
    return pure_static_raw_ids
