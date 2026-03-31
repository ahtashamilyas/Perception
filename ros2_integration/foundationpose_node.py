#!/usr/bin/env python3

"""
ROS2 FoundationPose Integration Node

This node provides a ROS2 interface to the FoundationPose system.
It subscribes to camera images and publishes pose estimations.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from vision_msgs.msg import Detection3D, Detection3DArray
from std_msgs.msg import Header
from cv_bridge import CvBridge

import numpy as np
import cv2
import subprocess
import os
import sys
import json
import yaml
from pathlib import Path
import tempfile

class FoundationPoseNode(Node):
    def __init__(self):
        super().__init__('foundationpose_node')
        
        # Declare parameters
        self.declare_parameter('foundationpose_path', '/home/student/Desktop/Perception/FoundationPose')
        self.declare_parameter('mesh_file', 'demo_data/mustard0/mesh/textured_simple.obj')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('object_frame', 'object')
        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('debug', True)
        
        # Get parameters
        self.foundationpose_path = self.get_parameter('foundationpose_path').get_parameter_value().string_value
        self.mesh_file = self.get_parameter('mesh_file').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.object_frame = self.get_parameter('object_frame').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10)
            
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10)
        
        # Publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/foundationpose/object_pose',
            10)
            
        self.detection_pub = self.create_publisher(
            Detection3DArray,
            '/foundationpose/detections',
            10)
        
        # Internal state
        self.latest_color_image = None
        self.latest_depth_image = None
        self.camera_info = None
        self.first_detection = True
        self.last_pose = None
        
        # Create temporary directory for communication with FoundationPose
        self.temp_dir = tempfile.mkdtemp(prefix='foundationpose_ros2_')
        
        # Timer for processing
        self.timer = self.create_timer(1.0/self.publish_rate, self.process_frame)
        
        self.get_logger().info(f'FoundationPose ROS2 Node initialized')
        self.get_logger().info(f'FoundationPose path: {self.foundationpose_path}')
        self.get_logger().info(f'Mesh file: {self.mesh_file}')
        self.get_logger().info(f'Temp directory: {self.temp_dir}')

    def image_callback(self, msg):
        """Handle incoming color images"""
        try:
            self.latest_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')

    def depth_callback(self, msg):
        """Handle incoming depth images"""
        try:
            # Handle different depth encodings
            if msg.encoding == '16UC1':
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
                # Convert to meters if needed (assuming mm input)
                self.latest_depth_image = self.latest_depth_image.astype(np.float32) / 1000.0
            elif msg.encoding == '32FC1':
                self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            else:
                self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')

    def camera_info_callback(self, msg):
        """Handle camera info messages"""
        self.camera_info = msg

    def process_frame(self):
        """Process the latest frame with FoundationPose"""
        if (self.latest_color_image is None or 
            self.latest_depth_image is None or 
            self.camera_info is None):
            return
            
        try:
            # Save current images to temporary files
            color_path = os.path.join(self.temp_dir, 'color.png')
            depth_path = os.path.join(self.temp_dir, 'depth.png')
            
            cv2.imwrite(color_path, self.latest_color_image)
            
            # Save depth as 16-bit PNG (convert back to mm for saving)
            depth_mm = (self.latest_depth_image * 1000.0).astype(np.uint16)
            cv2.imwrite(depth_path, depth_mm)
            
            # Create camera intrinsics file
            K = np.array(self.camera_info.k).reshape(3, 3)
            intrinsics_path = os.path.join(self.temp_dir, 'intrinsics.txt')
            np.savetxt(intrinsics_path, K)
            
            # Run FoundationPose estimation
            pose = self.run_foundationpose_estimation(color_path, depth_path, intrinsics_path)
            
            if pose is not None:
                self.publish_pose(pose)
                self.last_pose = pose
                
        except Exception as e:
            self.get_logger().error(f'Error in process_frame: {e}')

    def run_foundationpose_estimation(self, color_path, depth_path, intrinsics_path):
        """Run FoundationPose estimation using subprocess"""
        try:
            # Create a custom script to run FoundationPose
            script_path = os.path.join(self.temp_dir, 'run_estimation.py')
            
            script_content = f"""
import sys
sys.path.append('{self.foundationpose_path}')

from estimater import *
from datareader import *
import numpy as np
import cv2
import trimesh

# Load data
color = cv2.imread('{color_path}')
depth_mm = cv2.imread('{depth_path}', cv2.IMREAD_UNCHANGED)
depth = depth_mm.astype(np.float32) / 1000.0  # Convert to meters
K = np.loadtxt('{intrinsics_path}')

# Load mesh
mesh_path = os.path.join('{self.foundationpose_path}', '{self.mesh_file}')
mesh = trimesh.load(mesh_path)

# Initialize FoundationPose
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(
    model_pts=mesh.vertices, 
    model_normals=mesh.vertex_normals, 
    mesh=mesh, 
    scorer=scorer, 
    refiner=refiner, 
    debug_dir='{self.temp_dir}', 
    debug=0, 
    glctx=glctx
)

# Create a simple mask (you might want to use a segmentation model here)
mask = np.ones((color.shape[0], color.shape[1]), dtype=bool)

# Run estimation
if {str(self.first_detection).lower()}:
    pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
else:
    # Load previous pose for tracking
    prev_pose_path = '{os.path.join(self.temp_dir, "last_pose.txt")}'
    if os.path.exists(prev_pose_path):
        # For tracking, you would need to implement the tracking functionality
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
    else:
        pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=5)

# Save pose
pose_path = '{os.path.join(self.temp_dir, "pose_result.txt")}'
np.savetxt(pose_path, pose.reshape(4, 4))

print("Estimation completed successfully")
"""
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Run the script using the FoundationPose virtual environment
            venv_python = os.path.join(self.foundationpose_path, 'venv', 'bin', 'python')
            
            result = subprocess.run(
                [venv_python, script_path],
                cwd=self.foundationpose_path,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                # Load the pose result
                pose_result_path = os.path.join(self.temp_dir, 'pose_result.txt')
                if os.path.exists(pose_result_path):
                    pose_matrix = np.loadtxt(pose_result_path)
                    self.first_detection = False
                    
                    # Save for next tracking iteration
                    last_pose_path = os.path.join(self.temp_dir, 'last_pose.txt')
                    np.savetxt(last_pose_path, pose_matrix)
                    
                    return pose_matrix
                else:
                    self.get_logger().error('Pose result file not found')
                    return None
            else:
                self.get_logger().error(f'FoundationPose estimation failed: {result.stderr}')
                return None
                
        except subprocess.TimeoutExpired:
            self.get_logger().error('FoundationPose estimation timed out')
            return None
        except Exception as e:
            self.get_logger().error(f'Error running FoundationPose: {e}')
            return None

    def publish_pose(self, pose_matrix):
        """Publish pose as ROS2 messages"""
        try:
            # Create PoseStamped message
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.camera_frame
            
            # Extract translation and rotation from 4x4 matrix
            translation = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(rotation_matrix)
            quaternion = r.as_quat()  # Returns [x, y, z, w]
            
            pose_msg.pose.position.x = float(translation[0])
            pose_msg.pose.position.y = float(translation[1])
            pose_msg.pose.position.z = float(translation[2])
            
            pose_msg.pose.orientation.x = float(quaternion[0])
            pose_msg.pose.orientation.y = float(quaternion[1])
            pose_msg.pose.orientation.z = float(quaternion[2])
            pose_msg.pose.orientation.w = float(quaternion[3])
            
            self.pose_pub.publish(pose_msg)
            
            # Also publish as Detection3D for compatibility
            detection_array = Detection3DArray()
            detection_array.header = pose_msg.header
            
            detection = Detection3D()
            detection.header = pose_msg.header
            detection.results = []  # Could add classification results here
            detection.bbox.center = pose_msg.pose
            # You could set bbox.size based on your object dimensions
            
            detection_array.detections = [detection]
            self.detection_pub.publish(detection_array)
            
            if self.debug:
                self.get_logger().info(f'Published pose: translation={translation}, quaternion={quaternion}')
                
        except Exception as e:
            self.get_logger().error(f'Error publishing pose: {e}')

    def __del__(self):
        """Cleanup temporary directory"""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def main(args=None):
    rclpy.init(args=args)
    
    node = FoundationPoseNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
