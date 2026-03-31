#!/bin/bash

# ROS2-FoundationPose Integration Setup Script

echo "Setting up ROS2-FoundationPose Integration..."

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Check if required packages are installed
echo "Checking ROS2 packages..."

# Install required ROS2 packages if not present
sudo apt update
sudo apt install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-image-transport \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    python3-scipy

echo "Setup complete!"
echo ""
echo "To run the FoundationPose ROS2 bridge:"
echo "1. In one terminal:"
echo "   source /opt/ros/jazzy/setup.bash"
echo "   python3 /home/student/Desktop/Perception/FoundationPose/ros2_integration/ros2_bridge.py"
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
