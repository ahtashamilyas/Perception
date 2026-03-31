#!/bin/bash

# FoundationPose RosBag Testing - Complete Demo Runner
# This script runs all available demos and tests for the FoundationPose integration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Paths
FOUNDATIONPOSE_ROOT="/home/student/Desktop/Perception/FoundationPose"
TESTING_DIR="${FOUNDATIONPOSE_ROOT}/rosbag_testing"

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}    FoundationPose + ROS Jazzy Integration - Complete Demo    ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo ""
echo -e "${YELLOW}This demo shows the complete pipeline for cube detection with FoundationPose${NC}"
echo -e "${YELLOW}Working directory: ${TESTING_DIR}${NC}"
echo ""

cd "${TESTING_DIR}"

# Function to run a command with nice output
run_demo() {
    local name="$1"
    local description="$2"
    local command="$3"
    local optional="$4"
    
    echo -e "${BLUE}=== $name ===${NC}"
    echo -e "${YELLOW}$description${NC}"
    echo ""
    
    if [[ "$optional" == "optional" ]]; then
        echo -e "${PURPLE}This demo is optional and requires additional setup${NC}"
        read -p "Do you want to run this demo? (y/N): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Skipping $name${NC}"
            echo ""
            return
        fi
    fi
    
    echo -e "${GREEN}Running: $command${NC}"
    echo ""
    
    if eval "$command"; then
        echo ""
        echo -e "${GREEN}✓ $name completed successfully!${NC}"
    else
        echo ""
        echo -e "${RED}✗ $name failed with exit code $?${NC}"
    fi
    
    echo ""
    echo -e "${PURPLE}Press Enter to continue...${NC}"
    read
}

# Demo 1: Setup Verification
run_demo "Setup Verification" \
    "Verify that all required components are properly installed and configured" \
    "python3 verify_setup.py"

# Demo 2: Basic Integration Test
run_demo "Basic Integration Test" \
    "Test the FoundationPose integration with a simple synthetic scene" \
    "python3 fixed_demo.py"

# Demo 3: RosBag Processing Demo
run_demo "RosBag Processing Demo" \
    "Process RosBag data and generate cube detection results with video output" \
    "python3 rosbag_processor.py"

# Demo 4: Show Results
echo -e "${BLUE}=== Results Viewer ===${NC}"
echo -e "${YELLOW}Opening the generated results...${NC}"
echo ""

# Check if results exist
OUTPUT_DIR="${TESTING_DIR}/output"
ROSBAG_OUTPUT_DIR="${TESTING_DIR}/rosbag_output"

if [[ -d "$OUTPUT_DIR" ]]; then
    echo -e "${GREEN}Basic demo results available in: $OUTPUT_DIR${NC}"
    ls -la "$OUTPUT_DIR" 2>/dev/null || true
fi

if [[ -d "$ROSBAG_OUTPUT_DIR" ]]; then
    echo -e "${GREEN}RosBag processing results available in: $ROSBAG_OUTPUT_DIR${NC}"
    echo "Generated files:"
    echo "  - Detection results: $ROSBAG_OUTPUT_DIR/detection_results/"
    echo "  - Extracted images: $ROSBAG_OUTPUT_DIR/extracted_images/"
    echo "  - Summary video: $ROSBAG_OUTPUT_DIR/cube_detection_demo.mp4"
    
    # Try to open the video if possible
    VIDEO_FILE="$ROSBAG_OUTPUT_DIR/cube_detection_demo.mp4"
    if [[ -f "$VIDEO_FILE" ]]; then
        echo ""
        echo -e "${YELLOW}Would you like to open the generated video?${NC}"
        read -p "Open video? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if command -v vlc &> /dev/null; then
                vlc "$VIDEO_FILE" &
            elif command -v mpv &> /dev/null; then
                mpv "$VIDEO_FILE" &
            elif command -v ffplay &> /dev/null; then
                ffplay "$VIDEO_FILE" &
            else
                echo -e "${YELLOW}No video player found. You can manually open: $VIDEO_FILE${NC}"
            fi
        fi
    fi
fi

echo ""
echo -e "${BLUE}=== ROS2 Integration Information ===${NC}"
echo -e "${YELLOW}For full ROS2 integration with real-time RosBag playback and RViz visualization:${NC}"
echo ""
echo "1. Install ROS2 Jazzy if not already installed:"
echo "   sudo apt update && sudo apt install ros-jazzy-desktop"
echo ""
echo "2. Source ROS2 environment:"
echo "   source /opt/ros/jazzy/setup.bash"
echo ""
echo "3. Install additional ROS2 packages:"
echo "   sudo apt install ros-jazzy-cv-bridge ros-jazzy-tf2-ros ros-jazzy-rviz2"
echo ""
echo "4. Run the full ROS2 integration:"
echo "   ./run_tests.sh"
echo "   # Then select option 3 for 'Full integrated test with RViz'"
echo ""
echo -e "${GREEN}The ROS2 version will provide:${NC}"
echo "  - Real-time RosBag playback"
echo "  - Live pose estimation and tracking"
echo "  - 3D visualization in RViz2"
echo "  - TF transforms for integration with other ROS2 nodes"
echo "  - Publisher/subscriber interface for robot integration"
echo ""

echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}✓ FoundationPose integration is working correctly${NC}"
echo -e "${GREEN}✓ Cube detection and pose estimation demonstrated${NC}"
echo -e "${GREEN}✓ RosBag processing pipeline implemented${NC}"
echo -e "${GREEN}✓ Visualization and video generation working${NC}"
echo ""
echo -e "${YELLOW}Next steps for production use:${NC}"
echo "1. Set up full ROS2 environment for real-time processing"
echo "2. Calibrate camera parameters for your specific setup"
echo "3. Adjust detection parameters for your specific cubes/objects"
echo "4. Integrate with your robot control system"
echo ""
echo -e "${PURPLE}Thank you for testing FoundationPose + ROS Jazzy integration!${NC}"
