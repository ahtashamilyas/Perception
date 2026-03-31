#!/usr/bin/env python3

"""
Simple ROS2-FoundationPose Bridge Script

This script creates a bridge between ROS2 and your existing FoundationPose installation.
It runs in ROS2 Jazzy environment and communicates with FoundationPose via files.
"""

import os
import sys
import subprocess
import tempfile
import time
import json
import numpy as np

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def create_ros2_bridge():
    """Create the main ROS2 bridge script"""
    
    bridge_script = """#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import json
import subprocess
import tempfile
from scipy.spatial.transform import Rotation as R

class FoundationPoseBridge(Node):
    def __init__(self):
        super().__init__('foundationpose_bridge')
        
        # Parameters
        self.declare_parameter('foundationpose_path', '/home/student/Desktop/Perception/FoundationPose')
        self.declare_parameter('bridge_dir', '/tmp/foundationpose_bridge')
        self.declare_parameter('mesh_file', 'demo_data/mustard0/mesh/textured_simple.obj')
        
        self.fp_path = self.get_parameter('foundationpose_path').value
        self.bridge_dir = self.get_parameter('bridge_dir').value
        self.mesh_file = self.get_parameter('mesh_file').value
        
        # Create bridge directory
        os.makedirs(self.bridge_dir, exist_ok=True)
        
        # Initialize
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_depth = None
        self.camera_info = None
        
        # Subscribers
        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
            
        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/foundationpose/pose', 10)
        
        # Timer
        self.timer = self.create_timer(0.5, self.process_data)  # 2 Hz
        
        self.get_logger().info('FoundationPose Bridge initialized')
        
    def color_callback(self, msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
    def depth_callback(self, msg):
        if msg.encoding == '16UC1':
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '16UC1')
        elif msg.encoding == '32FC1':
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            
    def info_callback(self, msg):
        self.camera_info = msg
        
    def process_data(self):
        if self.latest_color is None or self.latest_depth is None or self.camera_info is None:
            return
            
        try:
            # Save data for FoundationPose
            color_path = os.path.join(self.bridge_dir, 'color.png')
            depth_path = os.path.join(self.bridge_dir, 'depth.png') 
            info_path = os.path.join(self.bridge_dir, 'camera_info.json')
            
            cv2.imwrite(color_path, self.latest_color)
            cv2.imwrite(depth_path, self.latest_depth)
            
            # Save camera info
            K = np.array(self.camera_info.k).reshape(3, 3)
            camera_data = {
                'K': K.tolist(),
                'width': self.camera_info.width,
                'height': self.camera_info.height
            }
            with open(info_path, 'w') as f:
                json.dump(camera_data, f)
                
            # Run FoundationPose estimation
            pose = self.run_foundationpose()
            
            if pose is not None:
                self.publish_pose(pose)
                
        except Exception as e:
            self.get_logger().error(f'Error in process_data: {e}')
            
    def run_foundationpose(self):
        try:
            # Create processing script
            script_path = os.path.join(self.bridge_dir, 'process.py')
            script_content = f'''
import sys
sys.path.append("{self.fp_path}")

from estimater import *
from datareader import *
import numpy as np
import cv2
import trimesh
import json
import os

# Load input data
color = cv2.imread("{self.bridge_dir}/color.png")
depth = cv2.imread("{self.bridge_dir}/depth.png", cv2.IMREAD_UNCHANGED)
if depth.dtype == np.uint16:
    depth = depth.astype(np.float32) / 1000.0

with open("{self.bridge_dir}/camera_info.json", 'r') as f:
    camera_data = json.load(f)
    K = np.array(camera_data['K'])

# Load mesh
mesh_path = os.path.join("{self.fp_path}", "{self.mesh_file}")
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
    debug_dir="{self.bridge_dir}",
    debug=0,
    glctx=glctx
)

# Simple mask (you can improve this with segmentation)
mask = np.ones((color.shape[0], color.shape[1]), dtype=bool)

# Estimate pose
pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=3)

# Save result
result_path = "{self.bridge_dir}/pose_result.txt"
np.savetxt(result_path, pose.reshape(4, 4))
print("SUCCESS")
'''
            
            with open(script_path, 'w') as f:
                f.write(script_content)
                
            # Run with FoundationPose environment
            venv_python = os.path.join(self.fp_path, 'venv', 'bin', 'python')
            result = subprocess.run(
                [venv_python, script_path],
                cwd=self.fp_path,
                capture_output=True,
                text=True,
                timeout=20
            )
            
            if result.returncode == 0 and "SUCCESS" in result.stdout:
                result_path = os.path.join(self.bridge_dir, 'pose_result.txt')
                if os.path.exists(result_path):
                    pose_matrix = np.loadtxt(result_path)
                    return pose_matrix
                    
        except Exception as e:
            self.get_logger().error(f'FoundationPose execution error: {e}')
            
        return None
        
    def publish_pose(self, pose_matrix):
        try:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_link'
            
            # Extract translation and rotation
            translation = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            
            # Convert to quaternion
            r = R.from_matrix(rotation_matrix)
            quat = r.as_quat()  # [x, y, z, w]
            
            msg.pose.position.x = float(translation[0])
            msg.pose.position.y = float(translation[1]) 
            msg.pose.position.z = float(translation[2])
            
            msg.pose.orientation.x = float(quat[0])
            msg.pose.orientation.y = float(quat[1])
            msg.pose.orientation.z = float(quat[2])
            msg.pose.orientation.w = float(quat[3])
            
            self.pose_pub.publish(msg)
            self.get_logger().info(f'Published pose: {translation}')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing pose: {e}')

def main():
    rclpy.init()
    node = FoundationPoseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
    
    # Write the bridge script
    bridge_path = os.path.join(current_dir, 'ros2_bridge.py')
    with open(bridge_path, 'w') as f:
        f.write(bridge_script)
    
    # Make it executable
    os.chmod(bridge_path, 0o755)
    
    return bridge_path

def create_setup_script():
    """Create setup script for the ROS2 integration"""
    
    setup_script = f"""#!/bin/bash

# ROS2-FoundationPose Integration Setup Script

echo "Setting up ROS2-FoundationPose Integration..."

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Check if required packages are installed
echo "Checking ROS2 packages..."

# Install required ROS2 packages if not present
sudo apt update
sudo apt install -y \\
    ros-jazzy-cv-bridge \\
    ros-jazzy-vision-msgs \\
    ros-jazzy-image-transport \\
    ros-jazzy-sensor-msgs \\
    ros-jazzy-geometry-msgs \\
    python3-scipy

echo "Setup complete!"
echo ""
echo "To run the FoundationPose ROS2 bridge:"
echo "1. In one terminal:"
echo "   source /opt/ros/jazzy/setup.bash"
echo "   python3 {current_dir}/ros2_bridge.py"
echo ""
echo "2. In another terminal, publish camera data or run your camera node"
echo ""
echo "The bridge will subscribe to:"
echo "  - /camera/color/image_raw"
echo "  - /camera/depth/image_raw" 
echo "  - /camera/color/camera_info"
echo ""
echo "And publish poses to:"
echo "  - /foundationpose/pose"
"""

    setup_path = os.path.join(current_dir, 'setup.sh')
    with open(setup_path, 'w') as f:
        f.write(setup_script)
    
    os.chmod(setup_path, 0o755)
    return setup_path

def create_test_script():
    """Create a test script to verify the integration"""
    
    test_script = """#!/usr/bin/env python3

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
"""

    test_path = os.path.join(current_dir, 'test_publisher.py')
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    os.chmod(test_path, 0o755)
    return test_path

if __name__ == '__main__':
    print("Creating ROS2-FoundationPose Integration...")
    
    bridge_path = create_ros2_bridge()
    setup_path = create_setup_script()
    test_path = create_test_script()
    
    print(f"✓ Created ROS2 bridge: {bridge_path}")
    print(f"✓ Created setup script: {setup_path}")
    print(f"✓ Created test script: {test_path}")
    print()
    print("Next steps:")
    print(f"1. Run setup: {setup_path}")
    print(f"2. Run bridge: python3 {bridge_path}")
    print(f"3. Test with: python3 {test_path}")
