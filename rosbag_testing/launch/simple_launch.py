#!/usr/bin/env python3

"""
Simple launch script for RosBag FoundationPose testing
"""

import os
import subprocess
import sys
import time
import signal

def main():
    # Paths
    foundationpose_root = "/home/student/Desktop/Perception/FoundationPose"
    rosbag_path = f"{foundationpose_root}/demo_data/jonas_data"
    testing_dir = f"{foundationpose_root}/rosbag_testing"
    
    print("=== RosBag FoundationPose Testing ===")
    print(f"RosBag path: {rosbag_path}")
    print(f"Testing directory: {testing_dir}")
    
    # Store process PIDs for cleanup
    processes = []
    
    def cleanup(signum=None, frame=None):
        print("\nCleaning up processes...")
        for p in processes:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                try:
                    p.kill()
                except:
                    pass
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        print("Starting RosBag player...")
        rosbag_cmd = ["ros2", "bag", "play", rosbag_path, "--loop", "--rate", "0.5"]
        rosbag_process = subprocess.Popen(rosbag_cmd, cwd=rosbag_path)
        processes.append(rosbag_process)
        time.sleep(2)
        
        print("Starting static transforms...")
        tf1_cmd = ["ros2", "run", "tf2_ros", "static_transform_publisher",
                   "0", "0", "0", "0", "0", "0", "camera_link", "camera_color_frame"]
        tf1_process = subprocess.Popen(tf1_cmd)
        processes.append(tf1_process)
        
        tf2_cmd = ["ros2", "run", "tf2_ros", "static_transform_publisher",
                   "0", "0", "0", "-1.570796", "0", "-1.570796", 
                   "camera_color_frame", "camera_color_optical_frame"]
        tf2_process = subprocess.Popen(tf2_cmd)
        processes.append(tf2_process)
        time.sleep(1)
        
        print("Starting FoundationPose node...")
        fp_cmd = ["python3", "rosbag_foundationpose_node.py"]
        fp_process = subprocess.Popen(fp_cmd, cwd=testing_dir)
        processes.append(fp_process)
        time.sleep(3)
        
        print("Starting RViz...")
        rviz_cmd = ["rviz2", "-d", f"{testing_dir}/cube_detection.rviz"]
        rviz_process = subprocess.Popen(rviz_cmd)
        processes.append(rviz_process)
        
        print("\nAll processes started successfully!")
        print("Press Ctrl+C to stop all processes")
        
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any critical process died
            if rosbag_process.poll() is not None:
                print("RosBag process died, restarting...")
                rosbag_process = subprocess.Popen(rosbag_cmd, cwd=rosbag_path)
                processes.append(rosbag_process)
            
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"Error: {e}")
        cleanup()

if __name__ == "__main__":
    main()
