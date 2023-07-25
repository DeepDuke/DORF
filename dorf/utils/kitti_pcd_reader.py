#! /usr/bin/python2.7

from __future__ import print_function
import numpy as np 
from pypcd import pypcd

import rospy 
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

from dorf.utils.color_utils import MOVING_OBJECT_LABELS, LABEL_TO_NAME, get_label_color


class PCDReader:
    """Read pointcloud map of *.pcd 
    """
    def __init__(self, pcd_file_path):
        self.pcd_file_path = pcd_file_path 
    
    
    def read_pcd_file(self):
        # pc = pcl.load_XYZI(self.pcd_file_path)
        # pc = pcl.PointCloud_PointXYZI(self.pcd_file_path)
        pc = pypcd.PointCloud.from_path(self.pcd_file_path)

        print('type of pc --> ', type(pc))
        return pc
    
    
if __name__ == '__main__':
    rospy.init_node('read_pcd_map_node')
    pc_pub = rospy.Publisher('kitti_map', PointCloud2, queue_size=10)
    
    pcd_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_map/05_2350_to_2670_w_interval2_voxel_0.200000.pcd'
    # pcd_file_path = '/home/spacex/noetic_workspace/REMO/SemanticKittiMap/00_4390_to_4530_w_interval2_voxel_0.200000.pcd'
    
    
    reader = PCDReader(pcd_file_path)
    
    pc = reader.read_pcd_file()
    
    
    points = pc.pc_data # shape n*4 
    
    # np.savetxt('points.csv', points, delimiter=',')
    # for point in points:
    #     intensity = np.uint32(point[3])
    #     instance_id = intensity >> 16      # get upper half for instances
    #     semantic_label = intensity & 0xFFFF    
    #     if semantic_label > 250:
    #         print('Found moving object point --> {}, {}'.format(semantic_label, LABEL_TO_NAME[semantic_label]))
    
    
    # Convert intensity to RGBA
    vis_points = np.zeros((points.shape[0], 5))
    for idx, point in enumerate(points):
        intensity = np.uint32(point[3])
        instance_id = intensity >> 16      # get upper half for instances
        semantic_label = intensity & 0xFFFF    
        # if semantic_label in MOVING_OBJECT_LABELS: 
        #     print('Found moving obejct semantic label --> ', semantic_label) 
        rgba = get_label_color(semantic_label)
        point[3] = rgba
        
        vis_points[idx] = np.array([point[0], point[1], point[2], semantic_label, rgba])
        
    header = Header()
    header.frame_id = 'map'
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
              PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
              PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
              PointField(name='intensity', offset=12, datatype=PointField.UINT32, count=1),
              PointField(name='rgba', offset=16, datatype=PointField.UINT32, count=1)]
    
    pc_msg = point_cloud2.create_cloud(header, fields, vis_points)    
    
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pc_pub.publish(pc_msg)
        rate.sleep()
        rospy.loginfo('Publishing kitti map')    
        