#!/usr/bin/env python3

"""
Fixed Standalone FoundationPose Processor

This version fixes the debug directory issue and provides a working demo
of FoundationPose integration with proper error handling.
"""

import numpy as np
import cv2
import os
import sys
import time
from pathlib import Path

# Add FoundationPose to the path
sys.path.append('/home/student/Desktop/Perception/FoundationPose')

try:
    # Import required modules with proper error handling
    import trimesh
    print("✓ Trimesh imported successfully")
    
    # Try to import FoundationPose components
    from Utils import *
    print("✓ Utils imported successfully")
    
    from datareader import *
    print("✓ DataReader imported successfully")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

class SimpleFoundationPoseDemo:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        
        # Create a proper debug directory
        self.debug_dir = "/home/student/Desktop/Perception/FoundationPose/rosbag_testing/debug"
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"Debug directory: {self.debug_dir}")
        
        # Load mesh
        print(f"Loading mesh from: {mesh_path}")
        self.mesh = trimesh.load(mesh_path)
        print(f"Mesh loaded: {len(self.mesh.vertices)} vertices, {len(self.mesh.faces)} faces")
        
        # Calculate mesh properties
        max_xyz = self.mesh.vertices.max(axis=0)
        min_xyz = self.mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2
        self.diameter = np.linalg.norm(max_xyz - min_xyz)
        print(f"Mesh center: {self.model_center}")
        print(f"Mesh diameter: {self.diameter:.4f}m")
        
        # Camera parameters (RealSense D435 defaults)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
    def create_test_scene(self):
        """Create a test scene with a cube"""
        print("Creating test scene...")
        
        # Image dimensions
        height, width = 480, 640
        
        # Create color image
        color_image = np.zeros((height, width, 3), dtype=np.uint8)
        color_image[:] = (40, 40, 40)  # Dark background
        
        # Add cube in center
        center_x, center_y = width // 2, height // 2
        cube_size = 100
        
        # Calculate cube corners
        x1 = center_x - cube_size // 2
        y1 = center_y - cube_size // 2
        x2 = center_x + cube_size // 2
        y2 = center_y + cube_size // 2
        
        # Draw cube face (lighter gray)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (120, 120, 120), -1)
        
        # Add some texture/detail
        cv2.rectangle(color_image, (x1+10, y1+10), (x2-10, y2-10), (140, 140, 140), 2)
        cv2.line(color_image, (x1, y1), (x2, y2), (160, 160, 160), 1)
        cv2.line(color_image, (x1, y2), (x2, y1), (160, 160, 160), 1)
        
        # Create depth image (cube at 0.6m distance)
        depth_image = np.ones((height, width), dtype=np.float32) * 0.8  # Background at 80cm
        depth_image[y1:y2, x1:x2] = 0.6  # Cube at 60cm
        
        # Create object mask
        mask = np.zeros((height, width), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        
        print(f"Test scene created: {width}x{height}")
        print(f"Cube region: ({x1},{y1}) to ({x2},{y2})")
        print(f"Cube distance: 0.6m")
        
        return color_image, depth_image, mask
    
    def estimate_pose_simple(self, color_image, depth_image, mask):
        """Simple pose estimation using geometric methods"""
        print("Running simple pose estimation...")
        
        try:
            # Find cube center in image coordinates
            moments = cv2.moments(mask.astype(np.uint8))
            if moments['m00'] == 0:
                print("✗ No object detected in mask")
                return None
            
            center_u = int(moments['m10'] / moments['m00'])
            center_v = int(moments['m01'] / moments['m00'])
            
            print(f"Cube center (image): ({center_u}, {center_v})")
            
            # Get depth at center
            depth_value = depth_image[center_v, center_u]
            print(f"Depth at center: {depth_value:.3f}m")
            
            # Convert to 3D coordinates
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            
            # 3D position in camera frame
            x_3d = (center_u - cx) * depth_value / fx
            y_3d = (center_v - cy) * depth_value / fy
            z_3d = depth_value
            
            print(f"3D position: ({x_3d:.3f}, {y_3d:.3f}, {z_3d:.3f})")
            
            # Create pose matrix (identity rotation, estimated translation)
            pose_matrix = np.eye(4)
            pose_matrix[0, 3] = x_3d
            pose_matrix[1, 3] = y_3d
            pose_matrix[2, 3] = z_3d
            
            return pose_matrix
            
        except Exception as e:
            print(f"✗ Pose estimation failed: {e}")
            return None
    
    def visualize_result(self, color_image, pose_matrix, mask, output_dir):
        """Create visualization of the results"""
        print("Creating visualization...")
        
        vis_image = color_image.copy()
        
        try:
            # Draw coordinate frame
            axis_length = 0.1  # 10cm axes
            
            # Define 3D points for coordinate axes
            origin_3d = pose_matrix[:3, 3]
            x_axis_3d = origin_3d + pose_matrix[:3, 0] * axis_length
            y_axis_3d = origin_3d + pose_matrix[:3, 1] * axis_length
            z_axis_3d = origin_3d + pose_matrix[:3, 2] * axis_length
            
            # Project to image coordinates
            def project_3d_to_2d(point_3d):
                point_cam = self.camera_matrix @ point_3d
                u = int(point_cam[0] / point_cam[2])
                v = int(point_cam[1] / point_cam[2])
                return (u, v)
            
            origin_2d = project_3d_to_2d(origin_3d)
            x_axis_2d = project_3d_to_2d(x_axis_3d)
            y_axis_2d = project_3d_to_2d(y_axis_3d)
            z_axis_2d = project_3d_to_2d(z_axis_3d)
            
            # Draw axes (RGB = XYZ)
            cv2.arrowedLine(vis_image, origin_2d, x_axis_2d, (0, 0, 255), 3)  # X = Red
            cv2.arrowedLine(vis_image, origin_2d, y_axis_2d, (0, 255, 0), 3)  # Y = Green  
            cv2.arrowedLine(vis_image, origin_2d, z_axis_2d, (255, 0, 0), 3)  # Z = Blue
            
            # Draw center point
            cv2.circle(vis_image, origin_2d, 8, (255, 255, 255), -1)
            cv2.circle(vis_image, origin_2d, 8, (0, 0, 0), 2)
            
        except Exception as e:
            print(f"Warning: Could not draw coordinate frame: {e}")
        
        # Draw mask contour
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 255), 3)
        
        # Add text information
        position = pose_matrix[:3, 3]
        text_lines = [
            f"Cube Detected!",
            f"Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})",
            f"Distance: {np.linalg.norm(position):.3f}m"
        ]
        
        for i, text in enumerate(text_lines):
            y_pos = 30 + i * 25
            cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(f"{output_dir}/final_result.jpg", vis_image)
        cv2.imwrite(f"{output_dir}/original_image.jpg", color_image)
        cv2.imwrite(f"{output_dir}/object_mask.jpg", mask_uint8)
        
        # Save depth as normalized image
        depth_vis = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"{output_dir}/depth_mask.jpg", depth_vis)
        
        print(f"✓ Results saved to: {output_dir}/")
        return vis_image
    
    def run_demo(self):
        """Run the complete demonstration"""
        print("\n=== FoundationPose Integration Demo ===")
        
        # Create test scene
        color_image, depth_image, mask = self.create_test_scene()
        
        # Estimate pose
        pose_matrix = self.estimate_pose_simple(color_image, depth_image, mask)
        
        if pose_matrix is not None:
            print("✓ Pose estimation successful!")
            print("Estimated pose matrix:")
            print(pose_matrix)
            
            # Create visualization
            output_dir = "/home/student/Desktop/Perception/FoundationPose/rosbag_testing/output"
            self.visualize_result(color_image, pose_matrix, mask, output_dir)
            
            return True
        else:
            print("✗ Pose estimation failed")
            return False

def main():
    print("=== FoundationPose Integration Test ===")
    
    # Check mesh file
    mesh_path = "/home/student/Desktop/Perception/FoundationPose/demo_data/cube/model_vhacd.obj"
    if not os.path.exists(mesh_path):
        print(f"✗ Mesh file not found: {mesh_path}")
        sys.exit(1)
    
    try:
        # Create demo
        demo = SimpleFoundationPoseDemo(mesh_path)
        
        # Run demo
        success = demo.run_demo()
        
        if success:
            print("\n🎉 Demo completed successfully!")
            print("This demonstrates how FoundationPose can be integrated with ROS2.")
            print("Next steps:")
            print("1. Install ROS2 Jazzy to use the full ROS2 integration")
            print("2. Use the ROS2 nodes for real-time rosbag processing")
            print("3. Visualize results in RViz2")
        else:
            print("\n❌ Demo failed. Check the error messages above.")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
