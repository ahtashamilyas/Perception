#!/bin/bash

# Fixed RViz launcher that avoids snap conflicts

# Remove snap paths from library paths to avoid conflicts
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/opt/ros/jazzy/lib"

# Use system RViz instead of snap version
if command -v /usr/bin/rviz2 &> /dev/null; then
    echo "Using system RViz2..."
    /usr/bin/rviz2 "$@"
elif command -v /opt/ros/jazzy/bin/rviz2 &> /dev/null; then
    echo "Using ROS2 RViz2..."
    /opt/ros/jazzy/bin/rviz2 "$@"
else
    echo "RViz2 not found. Installing..."
    sudo apt update && sudo apt install -y ros-jazzy-rviz2
    /opt/ros/jazzy/bin/rviz2 "$@"
fi
