#!/usr/bin/env python3

"""
System-Compatible FoundationPose Node

This version works with the system ROS2 installation by using the system Python
instead of the virtual environment Python for ROS2 components.
"""

import os
import sys
import subprocess
import time

# Check if we're in a virtual environment and need to handle ROS2 separately
def check_ros_environment():
    """Check ROS2 availability and Python compatibility"""
    try:
        # Try importing ROS2 with system Python
        result = subprocess.run([
            'python3', '-c', 
            'import rclpy; import sensor_msgs; import geometry_msgs; print("ROS2 available")'
        ], capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=''))
        
        if result.returncode == 0:
            print("✓ ROS2 Python modules available with system Python")
            return True
        else:
            print(f"✗ ROS2 Python modules not available: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error checking ROS2: {e}")
        return False

def run_with_system_python():
    """Run the ROS2 node with system Python to avoid virtual env conflicts"""
    
    # Set up environment for system Python with ROS2
    env = os.environ.copy()
    
    # Remove virtual environment from PATH and PYTHONPATH
    if 'VIRTUAL_ENV' in env:
        venv_path = env['VIRTUAL_ENV']
        path_parts = env.get('PATH', '').split(':')
        path_parts = [p for p in path_parts if not p.startswith(venv_path)]
        env['PATH'] = ':'.join(path_parts)
    
    # Ensure ROS2 is sourced
    if '/opt/ros/jazzy/setup.bash' not in env.get('BASH_ENV', ''):
        subprocess.run(['bash', '-c', 'source /opt/ros/jazzy/setup.bash'], env=env)
    
    # Add FoundationPose to Python path
    foundationpose_root = "/home/student/Desktop/Perception/FoundationPose"
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = f"{foundationpose_root}:{current_pythonpath}"
    else:
        env['PYTHONPATH'] = foundationpose_root
    
    # Path to the ROS2 node script
    node_script = "/home/student/Desktop/Perception/FoundationPose/rosbag_testing/system_ros_node.py"
    
    # Run with system Python
    try:
        print(f"Starting FoundationPose node with system Python...")
        process = subprocess.Popen([
            '/usr/bin/python3', node_script
        ], env=env)
        
        return process
    except Exception as e:
        print(f"Error starting node: {e}")
        return None

if __name__ == "__main__":
    if check_ros_environment():
        process = run_with_system_python()
        if process:
            try:
                process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
    else:
        print("ROS2 not properly configured. Please run: source /opt/ros/jazzy/setup.bash")
