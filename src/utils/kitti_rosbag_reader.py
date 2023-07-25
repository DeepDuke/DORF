#! /usr/bin/python

from __future__ import print_function
import numpy as np 
import rosbag 
# import bagpy

import rospy 
# from remo.msg import node
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import tf
from tf.transformations import quaternion_from_matrix
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from src.utils.color_utils import MOVING_OBJECT_LABELS, LABEL_TO_NAME, get_label_color


class RosbagReader:
    """Read kitti rosbag
    """
    def __init__(self, bag_file_path):
        self._bag_file_path = bag_file_path
        
    def read_bag(self):
        bag = rosbag.Bag(self._bag_file_path)
        NODE_MSG = bag.read_messages('/node/combined/optimized')
        return NODE_MSG

    
def test_main1():
    rospy.init_node('test_rosbag_reader_node')
    scan_pub = rospy.Publisher("scan", PointCloud2, queue_size=10)
    tf_pub = tf.TransformBroadcaster()
    
    # bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/00_4390_to_4530_w_interval_2_node.bag'
    # bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/01_150_to_250_w_interval_1_node.bag'
    # bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/02_860_to_950_w_interval_2_node.bag'
    # bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/05_2350_to_2670_w_interval_2_node.bag'
    bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/semantic_kitti_data/bag/07_630_to_820_w_interval_2_node.bag'
    
    reader = RosbagReader(bag_file_path)
    NODE_MSG = reader.read_bag()
    # print('type of NODE_MSG: ', type(NODE_MSG))
    node_msg_list = list(NODE_MSG)
    print('node_msg_list: {} messages', len(node_msg_list))
    
    # print('node_msg_list[0]: ', type(node_msg_list[0]))
    # topic, msg, t = node_msg_list[0]
    # print('odom: ', msg.odom)
    # print('lidar.header: ', msg.lidar.header)
    # print('topic --> ', topic)
    # # print('msg --> ', type(msg))
    # print('t --> ', t)
    # print(MOVING_OBJECT_LABELS)
    
    rate = rospy.Rate(10)

    for idx, (topic, msg, t) in enumerate(node_msg_list):
        # Visualize scan on rviz
        # PointCloud2
        pc_msg = msg.lidar
        pc_msg.header.frame_id = 'base_link'
        pc_msg.header.stamp = rospy.Time.now()
        
        # tf
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = 'map'
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = msg.odom.position.x 
        tf_msg.transform.translation.y = msg.odom.position.y
        tf_msg.transform.translation.z = msg.odom.position.z
            
        tf_msg.transform.rotation.x = msg.odom.orientation.x
        tf_msg.transform.rotation.y = msg.odom.orientation.y
        tf_msg.transform.rotation.z = msg.odom.orientation.z
        tf_msg.transform.rotation.w = msg.odom.orientation.w 
        
        scan_pub.publish(pc_msg)
        tf_pub.sendTransformMessage(tf_msg)
        rospy.loginfo('publishing {}-th msg'.format(idx))
        rate.sleep()


def test_main2():
    rospy.init_node('test_rosbag_reader_node')
    scan_pub = rospy.Publisher("scan", PointCloud2, queue_size=10)
    
    bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/bag/05_2350_to_2670_w_interval_2_node.bag'
    
    reader = RosbagReader(bag_file_path)
    NODE_MSG = reader.read_bag()
    # print('type of NODE_MSG: ', type(NODE_MSG))
    node_msg_list = list(NODE_MSG)
    print('node_msg_list: {} messages', len(node_msg_list))
    
    # print('node_msg_list[0]: ', type(node_msg_list[0]))
    # topic, msg, t = node_msg_list[0]
    # print('odom: ', msg.odom)
    # print('lidar.header: ', msg.lidar.header)
    # print('topic --> ', topic)
    # # print('msg --> ', type(msg))
    # print('t --> ', t)
    # print(MOVING_OBJECT_LABELS)
    
    rate = rospy.Rate(10)

    for idx, (topic, msg, t) in enumerate(node_msg_list):
        # Visualize scan on rviz
        # PointCloud2
        pc_msg = msg.lidar
        pc_msg.header.frame_id = 'map'
        pc_msg.header.stamp = rospy.Time.now()
        
        # tf
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = 'map'
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.child_frame_id = 'base_link'
        tf_msg.transform.translation.x = msg.odom.position.x 
        tf_msg.transform.translation.y = msg.odom.position.y
        tf_msg.transform.translation.z = msg.odom.position.z
            
        tf_msg.transform.rotation.x = msg.odom.orientation.x
        tf_msg.transform.rotation.y = msg.odom.orientation.y
        tf_msg.transform.rotation.z = msg.odom.orientation.z
        tf_msg.transform.rotation.w = msg.odom.orientation.w 
        
        # Do transform
        transformed_pc_msg = do_transform_cloud(pc_msg, tf_msg)
        scan_pub.publish(transformed_pc_msg)

        rospy.loginfo('publishing {}-th msg'.format(idx))
        rate.sleep()
        
def test_main3():
    rospy.init_node('test_rosbag_reader_node')
    scan_pub = rospy.Publisher("scan", PointCloud2, queue_size=10)
    
    bag_file_path = '/home/spacex/melodic_workspace/IROS2023/REMO_WS/src/REMO/bag/05_2350_to_2670_w_interval_2_node.bag'
    
    reader = RosbagReader(bag_file_path)
    NODE_MSG = reader.read_bag()
    # print('type of NODE_MSG: ', type(NODE_MSG))
    node_msg_list = list(NODE_MSG)
    print('node_msg_list: {} messages', len(node_msg_list))
    
    # print('node_msg_list[0]: ', type(node_msg_list[0]))
    # topic, msg, t = node_msg_list[0]
    # print('odom: ', msg.odom)
    # print('lidar.header: ', msg.lidar.header)
    # print('topic --> ', topic)
    # # print('msg --> ', type(msg))
    # print('t --> ', t)
    # print(MOVING_OBJECT_LABELS)
    
    rate = rospy.Rate(10)

    topic, msg, t = node_msg_list[0]
    # Visualize scan on rviz
    # PointCloud2
    pc_msg = msg.lidar
    pc_msg.header.frame_id = 'map'
    pc_msg.header.stamp = rospy.Time.now()
    
    # tf
    tf_msg = TransformStamped()
    tf_msg.header.frame_id = 'map'
    tf_msg.header.stamp = rospy.Time.now()
    tf_msg.child_frame_id = 'base_link'
    tf_msg.transform.translation.x = msg.odom.position.x 
    tf_msg.transform.translation.y = msg.odom.position.y
    tf_msg.transform.translation.z = msg.odom.position.z
        
    tf_msg.transform.rotation.x = msg.odom.orientation.x
    tf_msg.transform.rotation.y = msg.odom.orientation.y
    tf_msg.transform.rotation.z = msg.odom.orientation.z
    tf_msg.transform.rotation.w = msg.odom.orientation.w 
    
    # Do transform
    transformed_pc_msg = do_transform_cloud(pc_msg, tf_msg)
    while not rospy.is_shutdown():
        scan_pub.publish(transformed_pc_msg)
        rospy.loginfo('publishing {}-th msg'.format(0))
        rate.sleep()
    
if __name__ == '__main__':
    test_main1()