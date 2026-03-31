#!/usr/bin/env python3

"""
Quick verification script for RosBag FoundationPose setup
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} (NOT FOUND)")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists"""
    if os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} (NOT FOUND)")
        return False

def check_python_import(module, description):
    """Check if a Python module can be imported"""
    try:
        __import__(module)
        print(f"✓ {description}: {module}")
        return True
    except ImportError:
        print(f"✗ {description}: {module} (IMPORT FAILED)")
        return False

def main():
    print("=== RosBag FoundationPose Setup Verification ===\n")
    
    foundationpose_root = "/home/student/Desktop/Perception/FoundationPose"
    rosbag_testing_dir = f"{foundationpose_root}/rosbag_testing"
    
    all_good = True
    
    print("1. Checking core directories...")
    all_good &= check_directory(foundationpose_root, "FoundationPose root")
    all_good &= check_directory(rosbag_testing_dir, "RosBag testing directory")
    all_good &= check_directory(f"{foundationpose_root}/demo_data/jonas_data", "RosBag data directory")
    all_good &= check_directory(f"{foundationpose_root}/demo_data/cube", "Cube mesh directory")
    
    print("\n2. Checking RosBag files...")
    all_good &= check_file(f"{foundationpose_root}/demo_data/jonas_data/rosbag2_2025_05_23-11_03_48_0.mcap", "RosBag data file")
    all_good &= check_file(f"{foundationpose_root}/demo_data/jonas_data/metadata.yaml", "RosBag metadata")
    
    print("\n3. Checking mesh files...")
    all_good &= check_file(f"{foundationpose_root}/demo_data/cube/model_vhacd.obj", "Cube mesh (primary)")
    all_good &= check_file(f"{foundationpose_root}/demo_data/cube/part0.obj", "Cube mesh (alternative)")
    
    print("\n4. Checking testing package files...")
    all_good &= check_file(f"{rosbag_testing_dir}/rosbag_foundationpose_node.py", "Main FoundationPose node")
    all_good &= check_file(f"{rosbag_testing_dir}/cube_detection.rviz", "RViz configuration")
    all_good &= check_file(f"{rosbag_testing_dir}/run_tests.sh", "Testing script")
    all_good &= check_file(f"{rosbag_testing_dir}/launch/simple_launch.py", "Simple launch script")
    all_good &= check_file(f"{rosbag_testing_dir}/README.md", "Documentation")
    
    print("\n5. Checking Python dependencies...")
    all_good &= check_python_import("numpy", "NumPy")
    all_good &= check_python_import("cv2", "OpenCV")
    all_good &= check_python_import("trimesh", "Trimesh")
    all_good &= check_python_import("rclpy", "ROS2 Python client")
    all_good &= check_python_import("sensor_msgs", "ROS2 sensor messages")
    all_good &= check_python_import("geometry_msgs", "ROS2 geometry messages")
    all_good &= check_python_import("visualization_msgs", "ROS2 visualization messages")
    all_good &= check_python_import("cv_bridge", "ROS2 CV bridge")
    all_good &= check_python_import("tf2_ros", "ROS2 TF2")
    
    print("\n6. Checking FoundationPose modules...")
    sys.path.insert(0, foundationpose_root)
    try:
        import estimater
        print("✓ FoundationPose estimater module")
    except ImportError as e:
        print(f"✗ FoundationPose estimater module (IMPORT FAILED): {e}")
        all_good = False
    
    try:
        import datareader
        print("✓ FoundationPose datareader module")
    except ImportError as e:
        print(f"✗ FoundationPose datareader module (IMPORT FAILED): {e}")
        all_good = False
    
    print("\n7. Checking file permissions...")
    executable_files = [
        f"{rosbag_testing_dir}/rosbag_foundationpose_node.py",
        f"{rosbag_testing_dir}/run_tests.sh",
        f"{rosbag_testing_dir}/launch/simple_launch.py"
    ]
    
    for filepath in executable_files:
        if os.access(filepath, os.X_OK):
            print(f"✓ Executable: {os.path.basename(filepath)}")
        else:
            print(f"✗ Not executable: {os.path.basename(filepath)}")
            all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("🎉 ALL CHECKS PASSED! You're ready to run the tests.")
        print("\nNext steps:")
        print("1. cd /home/student/Desktop/Perception/FoundationPose/rosbag_testing")
        print("2. ./run_tests.sh")
        print("3. Select option 3 for 'Full integrated test with RViz'")
    else:
        print("❌ SOME CHECKS FAILED! Please fix the issues above before proceeding.")
        print("\nCommon solutions:")
        print("- Install missing Python packages: pip install <package_name>")
        print("- Check file paths and ensure all files are in correct locations")
        print("- Make scripts executable: chmod +x *.py *.sh")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
