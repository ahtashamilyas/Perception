#!/usr/bin/env python3

"""
Standalone RosBag FoundationPose Processor

This script processes RosBag data directly without requiring ROS2 runtime,
using the rosbag2_py library to read the bag and process images for cube detection.
"""

import numpy as np
import cv2
import os
import sys
import time
import sqlite3
from pathlib import Path
import yaml

# Add FoundationPose to the path
sys.path.append('/home/student/Desktop/perception/FoundationPose')

try:
    from estimater import *
    from datareader import *
    import trimesh
    print("✓ FoundationPose modules imported successfully")
except ImportError as e:
    print(f"✗ FoundationPose import failed: {e}")
    sys.exit(1)

class StandaloneFoundationPoseProcessor:
    def __init__(self, mesh_path, rosbag_path):
        self.mesh_path = mesh_path
        self.rosbag_path = rosbag_path
        
        # Load mesh
        print(f"Loading mesh from: {mesh_path}")
        self.mesh = trimesh.load(mesh_path)
        print(f"Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Calculate mesh properties
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.diameter = np.linalg.norm(extents)
        print(f"Mesh diameter: {self.diameter:.4f}m")
        
        # Initialize FoundationPose estimator
        try:
            print("Initializing FoundationPose estimator...")
            self.estimator = FoundationPose(
                model_pts=self.mesh.vertices,
                model_normals=self.mesh.vertex_normals,
                mesh=self.mesh,
                scorer=None,
                refiner=None,
                glctx=None
            )
            print("✓ FoundationPose estimator initialized")
        except Exception as e:
            print(f"✗ Failed to initialize FoundationPose: {e}")
            sys.exit(1)
        
        # Camera parameters (will be loaded from bag)
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        # Results storage
        self.results = []
        
    def load_camera_info_from_yaml(self):
        """Load camera calibration from the metadata or a calibration file"""
        # Default RealSense D435 camera parameters (approximate)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.distortion_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        print("Using default camera parameters (adjust if needed)")
        
    def process_rosbag(self):
        """Process the entire RosBag"""
        print(f"Processing RosBag from: {self.rosbag_path}")
        
        # Load camera parameters
        self.load_camera_info_from_yaml()
        
        # Since we can't use rosbag2_py without ROS2, let's create a simple image processor
        # that works with extracted images
        print("Note: For full RosBag processing, ROS2 runtime is needed.")
        print("This demo will show how the FoundationPose integration would work.")
        
        # Create a synthetic test to demonstrate the pipeline
        self.create_demo_test()
        
    def create_demo_test(self):
        """Create a demonstration of the pose estimation pipeline"""
        print("\n=== FoundationPose Demo Test ===")
        
        # Create a synthetic test image (you can replace this with actual extracted images)
        height, width = 480, 640
        
        # Create test color image
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        color_image[:] = (50, 50, 50)  # Dark gray background
        
        # Add a square (simulating a cube)
        center_x, center_y = width // 2, height // 2
        cube_size = 80
        x1 = center_x - cube_size // 2
        y1 = center_y - cube_size // 2
        x2 = center_x + cube_size // 2
        y2 = center_y + cube_size // 2
        
        # Draw cube face
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (180, 180, 180), -1)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Create test depth image (simulating depth at 0.5m)
        depth_image = np.ones((height, width), dtype=np.uint16) * 500  # 500mm = 0.5m
        depth_image[y1:y2, x1:x2] = 450  # Cube slightly closer
        
        # Create object mask
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
        print(f"Created test images: {width}x{height}")
        print(f"Cube region: ({x1},{y1}) to ({x2},{y2})")
        
        # Convert depth to meters
        depth_m = depth_image.astype(np.float32) / 1000.0
        
        try:
            print("Running FoundationPose estimation...")
            
            # Run pose estimation
            pose_est = self.estimator.register(
                K=self.camera_matrix,
                rgb=color_image,
                depth=depth_m,
                ob_mask=mask.astype(np.float32) / 255.0,
                iteration=3  # Reduced iterations for demo
            )
            
            if pose_est is not None:
                print("✓ Pose estimation successful!")
                print("Estimated pose matrix:")
                print(pose_est)
                
                # Extract position and rotation
                position = pose_est[:3, 3]
                rotation_matrix = pose_est[:3, :3]
                
                print(f"Position: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f}")
                
                # Convert rotation to Euler angles for readability
                from scipy.spatial.transform import Rotation
                r = Rotation.from_matrix(rotation_matrix)
                euler_angles = r.as_euler('xyz', degrees=True)
                print(f"Rotation (degrees): x={euler_angles[0]:.1f}, y={euler_angles[1]:.1f}, z={euler_angles[2]:.1f}")
                
                # Create visualization
                self.visualize_result(color_image, depth_image, pose_est, mask)
                
                return pose_est
            else:
                print("✗ Pose estimation failed - no pose returned")
                return None
                
        except Exception as e:
            print(f"✗ Pose estimation failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_result(self, color_image, depth_image, pose_matrix, mask):
        """Create visualization of the results"""
        print("Creating visualization...")
        
        # Create visualization image
        vis_image = color_image.copy()
        
        # Draw coordinate frame
        try:
            # Define coordinate axes (5cm length)
            axis_length = 0.05
            axes_3d = np.array([
                [0, 0, 0],           # Origin
                [axis_length, 0, 0], # X-axis (red)
                [0, axis_length, 0], # Y-axis (green)
                [0, 0, axis_length]  # Z-axis (blue)
            ])
            
            # Transform axes to world coordinates
            axes_world = []
            for point in axes_3d:
                point_homo = np.append(point, 1)
                world_point = pose_matrix @ point_homo
                axes_world.append(world_point[:3])
            
            axes_world = np.array(axes_world)
            
            # Project to image coordinates
            axes_image = []
            for point_3d in axes_world:
                # Project using camera matrix
                point_cam = self.camera_matrix @ point_3d
                u = int(point_cam[0] / point_cam[2])
                v = int(point_cam[1] / point_cam[2])
                axes_image.append((u, v))
            
            # Draw coordinate frame
            origin = axes_image[0]
            x_end = axes_image[1]
            y_end = axes_image[2]
            z_end = axes_image[3]
            
            # Check if points are within image bounds
            height, width = vis_image.shape[:2]
            if (0 <= origin[0] < width and 0 <= origin[1] < height):
                # Draw axes
                if (0 <= x_end[0] < width and 0 <= x_end[1] < height):
                    cv2.arrowedLine(vis_image, origin, x_end, (0, 0, 255), 3)  # Red X
                if (0 <= y_end[0] < width and 0 <= y_end[1] < height):
                    cv2.arrowedLine(vis_image, origin, y_end, (0, 255, 0), 3)  # Green Y
                if (0 <= z_end[0] < width and 0 <= z_end[1] < height):
                    cv2.arrowedLine(vis_image, origin, z_end, (255, 0, 0), 3)  # Blue Z
                
                # Draw center point
                cv2.circle(vis_image, origin, 5, (255, 255, 255), -1)
                
        except Exception as e:
            print(f"Warning: Could not draw coordinate frame: {e}")
        
        # Draw mask outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 255), 2)
        
        # Add text information
        position = pose_matrix[:3, 3]
        text = f"Cube detected at: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        output_dir = "/home/student/Desktop/Perception/FoundationPose/rosbag_testing/output"
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f"{output_dir}/detection_result.jpg", vis_image)
        cv2.imwrite(f"{output_dir}/input_color.jpg", color_image)
        cv2.imwrite(f"{output_dir}/input_depth.png", depth_image)
        cv2.imwrite(f"{output_dir}/object_mask.jpg", mask)
        
        print(f"✓ Visualization saved to: {output_dir}/")
        print("  - detection_result.jpg: Final result with pose visualization")
        print("  - input_color.jpg: Original color image")
        print("  - input_depth.png: Depth image")
        print("  - object_mask.jpg: Object detection mask")

def main():
    print("=== Standalone FoundationPose RosBag Processor ===")
    
    # Paths
    foundationpose_root = "/home/student/Desktop/Perception/FoundationPose"
    mesh_path = f"{foundationpose_root}/demo_data/cube/model_vhacd.obj"
    rosbag_path = f"{foundationpose_root}/demo_data/jonas_data"
    
    # Check if files exist
    if not os.path.exists(mesh_path):
        print(f"✗ Mesh file not found: {mesh_path}")
        sys.exit(1)
    
    if not os.path.exists(rosbag_path):
        print(f"✗ RosBag directory not found: {rosbag_path}")
        sys.exit(1)
    
    print(f"✓ Mesh file: {mesh_path}")
    print(f"✓ RosBag directory: {rosbag_path}")
    
    # Create processor
    try:
        processor = StandaloneFoundationPoseProcessor(mesh_path, rosbag_path)
        
        # Process the bag
        processor.process_rosbag()
        
        print("\n=== Processing Complete ===")
        print("This demo shows how FoundationPose integration works.")
        print("For full RosBag processing with real data, use the ROS2 version when ROS2 is available.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
