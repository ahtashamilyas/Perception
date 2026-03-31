#!/usr/bin/env python3

"""
RosBag FoundationPose Node - Cube Detection from Recorded RosBag

This node processes recorded RosBag data to detect and estimate the pose of cubes
using FoundationPose. It subscribes to camera topics from the bag and publishes
pose estimations and visualization markers.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
import tf2_ros

import numpy as np
import cv2
import os
import sys
import json
import yaml
from pathlib import Path
import tempfile
import threading
import queue
import time

# Add FoundationPose to the path
sys.path.append('/home/student/Desktop/Perception/FoundationPose')

try:
    from estimater import *
    from datareader import *
    import trimesh
except ImportError as e:
    print(f"Error importing FoundationPose modules: {e}")
    print("Please ensure FoundationPose is properly installed")


class RosBagFoundationPoseNode(Node):
    def __init__(self):
        super().__init__('rosbag_foundationpose_node')
        
        # Declare parameters
        self.declare_parameter('foundationpose_path', '/home/student/Desktop/Perception/FoundationPose')
        self.declare_parameter('mesh_file', 'demo_data/cube/model_vhacd.obj')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('object_frame', 'cube')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('debug', True)
        self.declare_parameter('confidence_threshold', 0.5)
        
        # Get parameters
        self.foundationpose_path = self.get_parameter('foundationpose_path').get_parameter_value().string_value
        self.mesh_file = self.get_parameter('mesh_file').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.object_frame = self.get_parameter('object_frame').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Initialize pose estimation queue
        self.estimation_queue = queue.Queue(maxsize=10)
        self.latest_camera_info = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # FoundationPose components
        self.estimator = None
        self.mesh = None
        self.to_origin = None
        self.diameter = None
        
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
        
        # Synchronization variables
        self.latest_color_image = None
        self.latest_depth_image = None
        self.latest_color_timestamp = None
        self.latest_depth_timestamp = None
        self.processing_lock = threading.Lock()
        
        # Timer for pose estimation
        self.estimation_timer = self.create_timer(1.0 / self.publish_rate, self.estimate_pose_callback)
        
        # Initialize FoundationPose
        self.initialize_foundationpose()
        
        self.get_logger().info("RosBag FoundationPose Node initialized")
        self.get_logger().info(f"Mesh file: {self.mesh_file}")
        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        self.get_logger().info(f"Object frame: {self.object_frame}")

    def initialize_foundationpose(self):
        """Initialize FoundationPose with cube mesh"""
        try:
            mesh_path = os.path.join(self.foundationpose_path, self.mesh_file)
            if not os.path.exists(mesh_path):
                self.get_logger().error(f"Mesh file not found: {mesh_path}")
                return False
                
            # Load mesh
            self.mesh = trimesh.load(mesh_path)
            self.get_logger().info(f"Loaded mesh from {mesh_path}")
            
            # Calculate mesh properties
            self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
            self.diameter = np.linalg.norm(extents)
            
            # Initialize estimator (will be done when camera info is available)
            self.get_logger().info("FoundationPose components initialized")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize FoundationPose: {e}")
            return False

    def camera_info_callback(self, msg):
        """Process camera info"""
        self.latest_camera_info = msg
        
        # Extract camera matrix
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)
        
        # Initialize estimator if not already done
        if self.estimator is None and self.mesh is not None:
            try:
                self.estimator = FoundationPose(
                    model_pts=self.mesh.vertices,
                    model_normals=self.mesh.vertex_normals,
                    mesh=self.mesh,
                    scorer=None,
                    refiner=None,
                    glctx=None
                )
                self.get_logger().info("FoundationPose estimator initialized")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize estimator: {e}")

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
                # Handle different depth encodings
                if msg.encoding == "16UC1":
                    self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                elif msg.encoding == "32FC1":
                    depth_32f = self.bridge.imgmsg_to_cv2(msg, "32FC1")
                    # Convert to 16UC1 (millimeters)
                    self.latest_depth_image = (depth_32f * 1000).astype(np.uint16)
                else:
                    self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                    
                self.latest_depth_timestamp = msg.header.stamp
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
                
            if self.estimator is None or self.camera_matrix is None:
                return
                
            # Perform pose estimation
            poses = self.estimate_pose(color_image, depth_image)
            
            if poses is not None and len(poses) > 0:
                # Publish results
                self.publish_poses(poses, timestamp)
                self.publish_markers(poses, timestamp)
                
                if self.debug:
                    debug_image = self.create_debug_image(color_image, poses)
                    self.publish_debug_image(debug_image, timestamp)
                    
        except Exception as e:
            self.get_logger().error(f"Error in pose estimation: {e}")

    def images_available(self):
        """Check if both color and depth images are available"""
        with self.processing_lock:
            return (self.latest_color_image is not None and 
                   self.latest_depth_image is not None and
                   self.latest_color_timestamp is not None and
                   self.latest_depth_timestamp is not None)

    def estimate_pose(self, color_image, depth_image):
        """Estimate pose using FoundationPose"""
        try:
            # Prepare inputs
            H, W = color_image.shape[:2]
            K = self.camera_matrix
            
            # Convert depth to meters if it's in millimeters
            if depth_image.dtype == np.uint16:
                depth_meters = depth_image.astype(np.float32) / 1000.0
            else:
                depth_meters = depth_image.astype(np.float32)
            
            # Run FoundationPose estimation
            pose_est = self.estimator.register(
                K=K, 
                rgb=color_image, 
                depth=depth_meters, 
                ob_mask=None,  # Will detect automatically
                iteration=5
            )
            
            if pose_est is not None:
                return [pose_est]
            else:
                return []
                
        except Exception as e:
            self.get_logger().error(f"Pose estimation failed: {e}")
            return []

    def publish_poses(self, poses, timestamp):
        """Publish pose estimations"""
        for i, pose_matrix in enumerate(poses):
            try:
                # Convert 4x4 transformation matrix to PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = timestamp
                pose_msg.header.frame_id = self.camera_frame
                
                # Extract position
                pose_msg.pose.position.x = float(pose_matrix[0, 3])
                pose_msg.pose.position.y = float(pose_matrix[1, 3])
                pose_msg.pose.position.z = float(pose_matrix[2, 3])
                
                # Extract rotation (convert to quaternion)
                rotation_matrix = pose_matrix[:3, :3]
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                pose_msg.pose.orientation.x = quaternion[0]
                pose_msg.pose.orientation.y = quaternion[1]
                pose_msg.pose.orientation.z = quaternion[2]
                pose_msg.pose.orientation.w = quaternion[3]
                
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
            
            # Set rotation
            rotation_matrix = pose_matrix[:3, :3]
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            transform.transform.rotation.x = quaternion[0]
            transform.transform.rotation.y = quaternion[1]
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {e}")

    def publish_markers(self, poses, timestamp):
        """Publish visualization markers"""
        try:
            marker_array = MarkerArray()
            
            for i, pose_matrix in enumerate(poses):
                # Cube marker
                cube_marker = Marker()
                cube_marker.header.frame_id = self.camera_frame
                cube_marker.header.stamp = timestamp
                cube_marker.ns = "cubes"
                cube_marker.id = i
                cube_marker.type = Marker.CUBE
                cube_marker.action = Marker.ADD
                
                # Set pose
                cube_marker.pose.position.x = float(pose_matrix[0, 3])
                cube_marker.pose.position.y = float(pose_matrix[1, 3])
                cube_marker.pose.position.z = float(pose_matrix[2, 3])
                
                rotation_matrix = pose_matrix[:3, :3]
                quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
                cube_marker.pose.orientation.x = quaternion[0]
                cube_marker.pose.orientation.y = quaternion[1]
                cube_marker.pose.orientation.z = quaternion[2]
                cube_marker.pose.orientation.w = quaternion[3]
                
                # Set scale (approximate cube size)
                cube_marker.scale.x = 0.05  # 5cm cube
                cube_marker.scale.y = 0.05
                cube_marker.scale.z = 0.05
                
                # Set color
                cube_marker.color.r = 1.0
                cube_marker.color.g = 0.0
                cube_marker.color.b = 0.0
                cube_marker.color.a = 0.7
                
                cube_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                
                marker_array.markers.append(cube_marker)
                
                # Coordinate frame marker
                frame_marker = Marker()
                frame_marker.header.frame_id = self.camera_frame
                frame_marker.header.stamp = timestamp
                frame_marker.ns = "frames"
                frame_marker.id = i
                frame_marker.type = Marker.ARROW
                frame_marker.action = Marker.ADD
                
                # Set pose (same as cube)
                frame_marker.pose = cube_marker.pose
                
                # Set scale for arrow
                frame_marker.scale.x = 0.1  # Length
                frame_marker.scale.y = 0.01  # Width
                frame_marker.scale.z = 0.01  # Height
                
                # Set color (blue for Z-axis)
                frame_marker.color.r = 0.0
                frame_marker.color.g = 0.0
                frame_marker.color.b = 1.0
                frame_marker.color.a = 1.0
                
                frame_marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()
                
                marker_array.markers.append(frame_marker)
            
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f"Error publishing markers: {e}")

    def create_debug_image(self, color_image, poses):
        """Create debug visualization image"""
        debug_image = color_image.copy()
        
        try:
            for pose_matrix in poses:
                # Project 3D cube vertices to 2D
                cube_vertices_3d = np.array([
                    [-0.025, -0.025, -0.025, 1],  # 2.5cm cube
                    [0.025, -0.025, -0.025, 1],
                    [0.025, 0.025, -0.025, 1],
                    [-0.025, 0.025, -0.025, 1],
                    [-0.025, -0.025, 0.025, 1],
                    [0.025, -0.025, 0.025, 1],
                    [0.025, 0.025, 0.025, 1],
                    [-0.025, 0.025, 0.025, 1]
                ]).T
                
                # Transform vertices
                vertices_cam = pose_matrix @ cube_vertices_3d
                
                # Project to image
                vertices_2d = self.camera_matrix @ vertices_cam[:3, :]
                vertices_2d = vertices_2d[:2, :] / vertices_2d[2, :]
                vertices_2d = vertices_2d.T.astype(np.int32)
                
                # Draw cube edges
                edges = [
                    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
                    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
                    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
                ]
                
                for edge in edges:
                    pt1 = tuple(vertices_2d[edge[0]])
                    pt2 = tuple(vertices_2d[edge[1]])
                    cv2.line(debug_image, pt1, pt2, (0, 255, 0), 2)
                
                # Draw coordinate frame
                origin = pose_matrix @ np.array([0, 0, 0, 1])[:, np.newaxis]
                x_axis = pose_matrix @ np.array([0.05, 0, 0, 1])[:, np.newaxis]
                y_axis = pose_matrix @ np.array([0, 0.05, 0, 1])[:, np.newaxis]
                z_axis = pose_matrix @ np.array([0, 0, 0.05, 1])[:, np.newaxis]
                
                # Project to image
                for axis, color in [(x_axis, (0, 0, 255)), (y_axis, (0, 255, 0)), (z_axis, (255, 0, 0))]:
                    origin_2d = self.camera_matrix @ origin[:3]
                    axis_2d = self.camera_matrix @ axis[:3]
                    
                    origin_2d = (origin_2d[:2] / origin_2d[2]).flatten().astype(np.int32)
                    axis_2d = (axis_2d[:2] / axis_2d[2]).flatten().astype(np.int32)
                    
                    cv2.arrowedLine(debug_image, tuple(origin_2d), tuple(axis_2d), color, 3)
                    
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

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = RosBagFoundationPoseNode()
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
