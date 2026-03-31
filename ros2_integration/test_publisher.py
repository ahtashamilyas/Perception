#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class TestPublisher(Node):
    def __init__(self):
        super().__init__('test_publisher')
        
        self.bridge = CvBridge()
        
        # Publishers
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        
        # Subscriber for results
        self.pose_sub = self.create_subscription(
            PoseStamped, '/foundationpose/pose', self.pose_callback, 10)
        
        # Timer to publish test data
        self.timer = self.create_timer(1.0, self.publish_test_data)
        
        self.get_logger().info('Test publisher started')
        
    def publish_test_data(self):
        # Create dummy test images
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        color[:] = (100, 150, 200)  # BGR
        
        depth = np.ones((480, 640), dtype=np.uint16) * 1000  # 1m depth
        
        # Convert to ROS messages
        color_msg = self.bridge.cv2_to_imgmsg(color, 'bgr8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth, '16UC1')
        
        # Camera info
        info_msg = CameraInfo()
        info_msg.header.stamp = self.get_clock().now().to_msg()
        info_msg.header.frame_id = 'camera_link'
        info_msg.width = 640
        info_msg.height = 480
        info_msg.k = [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]
        
        # Publish
        color_msg.header = info_msg.header
        depth_msg.header = info_msg.header
        
        self.color_pub.publish(color_msg)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info_msg)
        
    def pose_callback(self, msg):
        self.get_logger().info(f'Received pose: ({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')

def main():
    rclpy.init()
    node = TestPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
