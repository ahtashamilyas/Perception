#!/usr/bin/env python3

"""
Simplified RosBag FoundationPose Node for Cube Detection

This version uses classical computer vision methods for cube detection
and pose estimation, avoiding the need for complex neural network dependencies.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import tf2_ros

import numpy as np
import cv2
import os
import sys
import threading
import queue
import time

class SimplifiedFoundationPoseNode(Node):
    def __init__(self):
        super().__init__('simplified_foundationpose_node')
        
        # Declare parameters
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'cube')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('debug', True)
        self.declare_parameter('cube_size', 0.05)  # 5cm cube
        
        # Get parameters
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.object_frame = self.get_parameter('object_frame').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.cube_size = self.get_parameter('cube_size').get_parameter_value().double_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Image synchronization
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_color_timestamp = None
        self.processing_lock = threading.Lock()
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.color_image_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, 'cube_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'cube_markers', 10)
        self.debug_image_pub = self.create_publisher(Image, 'debug_image', 10)
        
        # Timer for pose estimation
        self.estimation_timer = self.create_timer(1.0 / self.publish_rate, self.estimate_pose_callback)
        
        # Object detection parameters
        self.hsv_lower = np.array([0, 0, 50])    # Lower HSV threshold for cube detection
        self.hsv_upper = np.array([180, 255, 255])  # Upper HSV threshold
        self.min_contour_area = 1000  # Minimum contour area
        
        self.get_logger().info("Simplified FoundationPose Node initialized")
        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        self.get_logger().info(f"Object frame: {self.object_frame}")
        self.get_logger().info(f"Cube size: {self.cube_size}m")

    def camera_info_callback(self, msg):
        """Process camera info"""
        # Extract camera matrix
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def color_image_callback(self, msg):
        """Process color image"""
        try:
            with self.processing_lock:
                self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.latest_color_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Error processing color image: {e}")

    def depth_image_callback(self, msg):
        """Process depth image"""
        try:
            with self.processing_lock:
                if msg.encoding == "16UC1":
                    self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                elif msg.encoding == "32FC1":
                    depth_32f = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                    self.latest_depth_image = (depth_32f * 1000).astype(np.uint16)
                else:
                    self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")

    def estimate_pose_callback(self):
        """Main pose estimation callback"""
        if not self.images_available():
            return
            
        try:
            with self.processing_lock:
                color_image = self.latest_color_image.copy()
                depth_image = self.latest_depth_image.copy()
                timestamp = self.latest_color_timestamp
                
            if self.camera_matrix is None:
                return
                
            # Detect cubes using computer vision
            cubes = self.detect_cubes(color_image, depth_image)
            
            if cubes:
                # Publish results
                self.publish_poses(cubes, timestamp)
                self.publish_markers(cubes, timestamp)
                
                if self.debug:
                    debug_image = self.create_debug_image(color_image, cubes)
                    self.publish_debug_image(debug_image, timestamp)
                    
        except Exception as e:
            self.get_logger().error(f"Error in pose estimation: {e}")

    def images_available(self):
        """Check if both color and depth images are available"""
        with self.processing_lock:
            return (self.latest_color_image is not None and 
                   self.latest_depth_image is not None and
                   self.latest_color_timestamp is not None)

    def detect_cubes(self, color_image, depth_image):
        """Detect cubes using classical computer vision"""
        cubes = []
        
        try:
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            
            # Create mask for object detection (you may need to adjust these values)
            mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
            
            # Morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's roughly square (cube-like)
                aspect_ratio = float(w) / h
                if 0.7 <= aspect_ratio <= 1.3:  # Roughly square
                    
                    # Get center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Get depth at center
                    if (0 <= center_y < depth_image.shape[0] and 
                        0 <= center_x < depth_image.shape[1]):
                        
                        depth_value = depth_image[center_y, center_x]
                        if depth_value > 0:  # Valid depth
                            
                            # Convert to 3D position
                            depth_m = depth_value / 1000.0  # Convert mm to meters
                            
                            # Use camera intrinsics to get 3D position
                            fx = self.camera_matrix[0, 0]
                            fy = self.camera_matrix[1, 1]
                            cx = self.camera_matrix[0, 2]
                            cy = self.camera_matrix[1, 2]
                            
                            # Calculate 3D position
                            x_3d = (center_x - cx) * depth_m / fx
                            y_3d = (center_y - cy) * depth_m / fy
                            z_3d = depth_m
                            
                            # Estimate orientation (simplified - assume upright cube)
                            # You could use PnP or other methods for better orientation estimation
                            pose_matrix = np.eye(4)
                            pose_matrix[0, 3] = x_3d
                            pose_matrix[1, 3] = y_3d
                            pose_matrix[2, 3] = z_3d
                            
                            cube_info = {
                                'pose_matrix': pose_matrix,
                                'center_2d': (center_x, center_y),
                                'bbox': (x, y, w, h),
                                'area': area,
                                'contour': contour
                            }
                            
                            cubes.append(cube_info)
                            
        except Exception as e:
            self.get_logger().error(f"Error in cube detection: {e}")
            
        return cubes

    def publish_poses(self, cubes, timestamp):
        """Publish pose estimations"""
        for i, cube in enumerate(cubes):
            try:
                pose_matrix = cube['pose_matrix']
                
                # Convert to PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = timestamp
                pose_msg.header.frame_id = self.camera_frame
                
                # Set position
                pose_msg.pose.position.x = float(pose_matrix[0, 3])
                pose_msg.pose.position.y = float(pose_matrix[1, 3])
                pose_msg.pose.position.z = float(pose_matrix[2, 3])
                
                # Set orientation (identity quaternion for now)
                pose_msg.pose.orientation.x = 0.0
                pose_msg.pose.orientation.y = 0.0
                pose_msg.pose.orientation.z = 0.0
                pose_msg.pose.orientation.w = 1.0
                
                self.pose_pub.publish(pose_msg)
                
                # Also publish TF
                self.publish_transform(pose_matrix, timestamp, f"{self.object_frame}_{i}")
                
            except Exception as e:
                self.get_logger().error(f"Error publishing pose {i}: {e}")

    def publish_transform(self, pose_matrix, timestamp, frame_id):
        """Publish TF transform"""
        try:
            transform = TransformStamped()
            transform.header.stamp = timestamp
            transform.header.frame_id = self.camera_frame
            transform.child_frame_id = frame_id
            
            # Set translation
            transform.transform.translation.x = float(pose_matrix[0, 3])
            transform.transform.translation.y = float(pose_matrix[1, 3])
            transform.transform.translation.z = float(pose_matrix[2, 3])
            
            # Set rotation (identity for now)
            transform.transform.rotation.x = 0.0
            transform.transform.rotation.y = 0.0
            transform.transform.rotation.z = 0.0
            transform.transform.rotation.w = 1.0
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {e}")

    def publish_markers(self, cubes, timestamp):
        """Publish visualization markers"""
        try:
            marker_array = MarkerArray()
            
            for i, cube in enumerate(cubes):
                pose_matrix = cube['pose_matrix']
                
                # Cube marker
                cube_marker = Marker()
                cube_marker.header.frame_id = self.camera_frame
                cube_marker.header.stamp = timestamp
                cube_marker.ns = "detected_cubes"
                cube_marker.id = i
                cube_marker.type = Marker.CUBE
                cube_marker.action = Marker.ADD
                
                # Set pose
                cube_marker.pose.position.x = float(pose_matrix[0, 3])
                cube_marker.pose.position.y = float(pose_matrix[1, 3])
                cube_marker.pose.position.z = float(pose_matrix[2, 3])
                
                cube_marker.pose.orientation.x = 0.0
                cube_marker.pose.orientation.y = 0.0
                cube_marker.pose.orientation.z = 0.0
                cube_marker.pose.orientation.w = 1.0
                
                # Set scale
                cube_marker.scale.x = self.cube_size
                cube_marker.scale.y = self.cube_size
                cube_marker.scale.z = self.cube_size
                
                # Set color (red with transparency)
                cube_marker.color.r = 1.0
                cube_marker.color.g = 0.0
                cube_marker.color.b = 0.0
                cube_marker.color.a = 0.7
                
                cube_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                
                marker_array.markers.append(cube_marker)
                
                # Text marker with information
                text_marker = Marker()
                text_marker.header.frame_id = self.camera_frame
                text_marker.header.stamp = timestamp
                text_marker.ns = "cube_labels"
                text_marker.id = i
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                
                # Position slightly above the cube
                text_marker.pose.position.x = float(pose_matrix[0, 3])
                text_marker.pose.position.y = float(pose_matrix[1, 3])
                text_marker.pose.position.z = float(pose_matrix[2, 3]) + self.cube_size
                
                text_marker.pose.orientation.w = 1.0
                
                # Set text
                text_marker.text = f"Cube {i}\nD: {pose_matrix[2, 3]:.2f}m"
                
                # Set scale
                text_marker.scale.z = 0.02
                
                # Set color (white)
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0
                
                text_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                
                marker_array.markers.append(text_marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing markers: {e}")

    def create_debug_image(self, color_image, cubes):
        """Create debug visualization image"""
        debug_image = color_image.copy()
        
        try:
            for i, cube in enumerate(cubes):
                # Draw bounding box
                x, y, w, h = cube['bbox']
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw center point
                center_x, center_y = cube['center_2d']
                cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
                
                # Draw contour
                cv2.drawContours(debug_image, [cube['contour']], -1, (255, 0, 0), 2)
                
                # Add text with 3D position
                pose_matrix = cube['pose_matrix']
                text = f"Cube {i}: ({pose_matrix[0,3]:.2f}, {pose_matrix[1,3]:.2f}, {pose_matrix[2,3]:.2f})"
                cv2.putText(debug_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        except Exception as e:
            self.get_logger().error(f"Error creating debug image: {e}")
            
        return debug_image

    def publish_debug_image(self, debug_image, timestamp):
        """Publish debug image"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header.stamp = timestamp
            debug_msg.header.frame_id = self.camera_frame
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = SimplifiedFoundationPoseNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
