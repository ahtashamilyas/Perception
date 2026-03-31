#!/bin/bash

# RosBag FoundationPose Testing Script
# This script provides different testing scenarios for FoundationPose with RosBag data

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
FOUNDATIONPOSE_ROOT="/home/student/Desktop/Perception/FoundationPose"
ROSBAG_TESTING_DIR="${FOUNDATIONPOSE_ROOT}/rosbag_testing"
ROSBAG_PATH="${FOUNDATIONPOSE_ROOT}/demo_data/jonas_data"
MESH_FILE="demo_data/cube/model_vhacd.obj"
MESH_DIRECTORY="demo_data"

echo -e "${BLUE}=== RosBag FoundationPose Testing Suite ===${NC}"
echo -e "${BLUE}Root Directory: ${FOUNDATIONPOSE_ROOT}${NC}"
echo -e "${BLUE}RosBag Path: ${ROSBAG_PATH}${NC}"
echo -e "${BLUE}Mesh File: ${MESH_FILE}${NC}"
echo -e "${BLUE}Mesh Directory: ${MESH_DIRECTORY}${NC}"

# Function to run multi-object detection test
run_multi_object_test() {
    echo -e "${BLUE}Running multi-object detection test...${NC}"
    echo -e "${YELLOW}This test will detect multiple objects in the rosbag video${NC}"
    
    # Check mesh directory
    check_mesh_directory
    
    # Export Python path
    export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:$PYTHONPATH"
    
    # Start rosbag
    echo -e "${BLUE}Starting RosBag player...${NC}"
    cd "${ROSBAG_PATH}"
    ros2 bag play . --loop --rate 0.3 &  # Moderate speed for multi-object processing
    ROSBAG_PID=$!
    
    sleep 3
    
    # Start transforms
    echo -e "${BLUE}Starting transforms...${NC}"
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 camera_link camera_color_frame &
    TF1_PID=$!
    ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 camera_color_frame camera_color_optical_frame &
    TF2_PID=$!
    
    sleep 2
    
    # Run FoundationPose in multi-object mode
    echo -e "${BLUE}Starting FoundationPose in multi-object mode...${NC}"
    cd "${ROSBAG_TESTING_DIR}"
    
    ros2 run system_ros_node system_foundationpose_node \
        --ros-args \
        -p multi_object_mode:=true \
        -p mesh_directory:="${MESH_DIRECTORY}" \
        -p camera_frame:=camera_color_optical_frame \
        -p object_frame:=multi_objects \
        -p debug:=true \
        -p track_refine_iter:=2 \
        -p est_refine_iter:=4 &
    FP_PID=$!
    
    sleep 3
    
    # Start RViz for multi-object visualization
    echo -e "${BLUE}Starting RViz for multi-object visualization...${NC}"
    "${ROSBAG_TESTING_DIR}/fixed_rviz.sh" -d "${ROSBAG_TESTING_DIR}/detection_visualization.rviz" &
    RVIZ_PID=$!
    
    echo ""
    echo -e "${GREEN}Multi-object detection started!${NC}"
    echo -e "${YELLOW}Expected behavior:${NC}"
    echo -e "  • Multiple colored cubes will be detected automatically"
    echo -e "  • Each object gets its own pose topic: /object_X_pose"
    echo -e "  • Markers will be published to: /object_markers"
    echo -e "  • Processing may be slower due to multiple objects"
    echo -e "${YELLOW}PIDs: RosBag=${ROSBAG_PID}, TF1=${TF1_PID}, TF2=${TF2_PID}, FoundationPose=${FP_PID}, RViz=${RVIZ_PID}${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"
    
    # Wait for user interrupt
    trap 'echo -e "\n${YELLOW}Stopping all processes...${NC}"; kill $ROSBAG_PID $TF1_PID $TF2_PID $FP_PID $RVIZ_PID 2>/dev/null; exit 0' INT
    wait
}ut
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
FOUNDATIONPOSE_ROOT="/home/student/Desktop/Perception/FoundationPose"
ROSBAG_TESTING_DIR="${FOUNDATIONPOSE_ROOT}/rosbag_testing"
ROSBAG_PATH="${FOUNDATIONPOSE_ROOT}/demo_data/jonas_data"
MESH_FILE="demo_data/cube/model_vhacd.obj"
MESH_DIRECTORY="demo_data"

echo -e "${BLUE}=== RosBag FoundationPose Testing Suite ===${NC}"
echo -e "${BLUE}Root Directory: ${FOUNDATIONPOSE_ROOT}${NC}"
echo -e "${BLUE}RosBag Path: ${ROSBAG_PATH}${NC}"
echo -e "${BLUE}Mesh File: ${MESH_FILE}${NC}"
echo -e "${BLUE}Mesh Directory: ${MESH_DIRECTORY}${NC}"

# Function to check if RosBag exists
check_rosbag() {
    if [ ! -f "${ROSBAG_PATH}/rosbag2_2025_05_23-11_03_48_0.mcap" ]; then
        echo -e "${RED}Error: RosBag file not found at ${ROSBAG_PATH}/rosbag2_2025_05_23-11_03_48_0.mcap${NC}"
        exit 1
    fi
    
    if [ ! -f "${ROSBAG_PATH}/metadata.yaml" ]; then
        echo -e "${RED}Error: RosBag metadata not found at ${ROSBAG_PATH}/metadata.yaml${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ RosBag files found${NC}"
}

# Function to check if mesh file exists
check_mesh() {
    if [ ! -f "${FOUNDATIONPOSE_ROOT}/${MESH_FILE}" ]; then
        echo -e "${RED}Error: Mesh file not found at ${FOUNDATIONPOSE_ROOT}/${MESH_FILE}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Mesh file found${NC}"
}

# Function to check for mesh directory
check_mesh_directory() {
    if [ ! -d "${FOUNDATIONPOSE_ROOT}/${MESH_DIRECTORY}" ]; then
        echo -e "${RED}Error: Mesh directory not found at ${FOUNDATIONPOSE_ROOT}/${MESH_DIRECTORY}${NC}"
        exit 1
    fi
    
    # Count mesh files
    mesh_count=$(find "${FOUNDATIONPOSE_ROOT}/${MESH_DIRECTORY}" -name "*.obj" -o -name "*.stl" -o -name "*.STL" | wc -l)
    echo -e "${GREEN}✓ Mesh directory found with ${mesh_count} mesh files${NC}"
}

# Function to make scripts executable
make_executable() {
    chmod +x "${ROSBAG_TESTING_DIR}/rosbag_foundationpose_node.py"
    chmod +x "${ROSBAG_TESTING_DIR}/launch_rosbag_test.py"
    echo -e "${GREEN}✓ Scripts made executable${NC}"
}

# Function to show available options
show_menu() {
    echo ""
    echo -e "${YELLOW}Available Testing Options:${NC}"
    echo "1. Play RosBag only (to inspect topics)"
    echo "2. Run FoundationPose node only (manual rosbag required)"
    echo "3. Full integrated test with RViz"
    echo "4. Debug mode with verbose output"
    echo "5. Multi-object detection test"
    echo "6. Check RosBag info"
    echo "7. Test individual components"
    echo "8. Exit"
    echo ""
}

# Function to play rosbag only
play_rosbag_only() {
    echo -e "${BLUE}Playing RosBag only...${NC}"
    echo -e "${YELLOW}In another terminal, you can run: ros2 topic list${NC}"
    echo -e "${YELLOW}To see image topics: ros2 run rqt_image_view rqt_image_view${NC}"
    
    cd "${ROSBAG_PATH}"
    ros2 bag play . --loop --rate 0.5
}

# Function to run FoundationPose node only
run_foundationpose_only() {
    echo -e "${BLUE}Running FoundationPose node only...${NC}"
    echo -e "${YELLOW}Make sure to play the rosbag in another terminal first!${NC}"
    echo -e "${YELLOW}Command: cd ${ROSBAG_PATH} && ros2 bag play . --loop --rate 0.5${NC}"
    
    read -p "Press Enter when rosbag is playing..."
    
    cd "${ROSBAG_TESTING_DIR}"
    /usr/bin/python3 system_ros_node.py
}

# Function to run full integrated test
run_full_test() {
    echo -e "${BLUE}Running full integrated test with RViz...${NC}"
    echo -e "${YELLOW}This will start:${NC}"
    echo -e "${YELLOW}  1. RosBag player (looping at 0.5x speed)${NC}"
    echo -e "${YELLOW}  2. FoundationPose detection node${NC}"
    echo -e "${YELLOW}  3. RViz for visualization${NC}"
    echo -e "${YELLOW}  4. TF static transforms${NC}"
    
    echo ""
    echo -e "${GREEN}Starting in 3 seconds... Press Ctrl+C to cancel${NC}"
    sleep 3
    
    # Start rosbag in background
    echo -e "${BLUE}Starting RosBag player...${NC}"
    cd "${ROSBAG_PATH}"
    ros2 bag play . --loop --rate 0.5 &
    ROSBAG_PID=$!
    
    # Wait a moment for rosbag to start
    sleep 2
    
    # Start static transform publishers
    echo -e "${BLUE}Starting static transforms...${NC}"
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 camera_link camera_color_frame &
    TF1_PID=$!
    
    ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 camera_color_frame camera_color_optical_frame &
    TF2_PID=$!
    
    # Wait a moment for transforms
    sleep 1
    
    # Start FoundationPose node in background
    echo -e "${BLUE}Starting FoundationPose node...${NC}"
    cd "${ROSBAG_TESTING_DIR}"
    # Use system Python to avoid virtual environment conflicts
    /usr/bin/python3 system_ros_node.py &
    FP_PID=$!
    
    # Wait a moment for node to initialize
    sleep 3
    
    # Start RViz
    echo -e "${BLUE}Starting RViz...${NC}"
    "${ROSBAG_TESTING_DIR}/fixed_rviz.sh" -d "${ROSBAG_TESTING_DIR}/detection_visualization.rviz" &
    RVIZ_PID=$!
    
    echo ""
    echo -e "${GREEN}All components started!${NC}"
    echo -e "${YELLOW}PIDs: RosBag=${ROSBAG_PID}, TF1=${TF1_PID}, TF2=${TF2_PID}, FoundationPose=${FP_PID}, RViz=${RVIZ_PID}${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"
    
    # Wait for user interrupt
    trap 'echo -e "\n${YELLOW}Stopping all processes...${NC}"; kill $ROSBAG_PID $TF1_PID $TF2_PID $FP_PID $RVIZ_PID 2>/dev/null; exit 0' INT
    wait
}

# Function to run debug mode
run_debug_mode() {
    echo -e "${BLUE}Running debug mode...${NC}"
    
    # Export Python path
    export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:$PYTHONPATH"
    
    # Start rosbag
    echo -e "${BLUE}Starting RosBag player...${NC}"
    cd "${ROSBAG_PATH}"
    ros2 bag play . --loop --rate 0.2 &  # Slower for debugging
    ROSBAG_PID=$!
    
    sleep 2
    
    # Start transforms
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 camera_link camera_color_frame &
    TF1_PID=$!
    ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 camera_color_frame camera_color_optical_frame &
    TF2_PID=$!
    
    sleep 1
    
    # Run FoundationPose with debug output
    echo -e "${BLUE}Starting FoundationPose in debug mode...${NC}"
    cd "${ROSBAG_TESTING_DIR}"
    
    # Set ROS2 logging to debug
    export RCUTILS_LOGGING_SEVERITY=DEBUG
    
    /usr/bin/python3 -u system_ros_node.py &
    FP_PID=$!
    
    echo ""
    echo -e "${GREEN}Debug mode started!${NC}"
    echo -e "${YELLOW}Monitor the output for detailed information${NC}"
    echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
    
    trap 'echo -e "\n${YELLOW}Stopping debug processes...${NC}"; kill $ROSBAG_PID $TF1_PID $TF2_PID $FP_PID 2>/dev/null; exit 0' INT
    wait
}

# Function to check rosbag info
check_rosbag_info() {
    echo -e "${BLUE}RosBag Information:${NC}"
    echo ""
    
    cd "${ROSBAG_PATH}"
    
    echo -e "${YELLOW}=== RosBag Info ===${NC}"
    ros2 bag info .
    
    echo ""
    echo -e "${YELLOW}=== Topics in bag ===${NC}"
    ros2 bag info . | grep -A 100 "Topic information:"
    
    echo ""
    echo -e "${YELLOW}=== Checking for required topics ===${NC}"
    
    # Check if required topics exist
    TOPICS=$(ros2 bag info . | grep "Topic:" | awk '{print $2}')
    
    for topic in "/camera/camera/color/image_raw" "/camera/camera/depth/image_rect_raw" "/camera/camera/color/camera_info"; do
        if echo "$TOPICS" | grep -q "$topic"; then
            echo -e "${GREEN}✓ Found: $topic${NC}"
        else
            echo -e "${RED}✗ Missing: $topic${NC}"
        fi
    done
}

# Function to test individual components
test_components() {
    echo -e "${BLUE}Testing individual components...${NC}"
    echo ""
    
    echo -e "${YELLOW}1. Testing Python imports...${NC}"
    cd "${FOUNDATIONPOSE_ROOT}"
    python3 -c "
import sys
sys.path.append('.')
try:
    import numpy as np
    print('✓ numpy imported successfully')
except Exception as e:
    print(f'✗ numpy import failed: {e}')

try:
    import cv2
    print('✓ cv2 imported successfully')
except Exception as e:
    print(f'✗ cv2 import failed: {e}')

try:
    import trimesh
    print('✓ trimesh imported successfully')
except Exception as e:
    print(f'✗ trimesh import failed: {e}')

try:
    from estimater import *
    print('✓ FoundationPose estimater imported successfully')
except Exception as e:
    print(f'✗ FoundationPose estimater import failed: {e}')

try:
    import rclpy
    print('✓ rclpy imported successfully')
except Exception as e:
    print(f'✗ rclpy import failed: {e}')
"
    
    echo ""
    echo -e "${YELLOW}2. Testing mesh file loading...${NC}"
    python3 -c "
import sys
sys.path.append('.')
try:
    import trimesh
    mesh = trimesh.load('${MESH_FILE}')
    print(f'✓ Mesh loaded successfully: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces')
    print(f'  Mesh bounds: {mesh.bounds}')
    print(f'  Mesh extents: {mesh.extents}')
except Exception as e:
    print(f'✗ Mesh loading failed: {e}')
"
    
    echo ""
    echo -e "${YELLOW}3. Testing ROS2 topics (requires running rosbag)...${NC}"
    echo "To test topics, start the rosbag and run:"
    echo "  ros2 topic list"
    echo "  ros2 topic echo /camera/camera/color/camera_info --once"
    echo "  ros2 topic hz /camera/camera/color/image_raw"
}

# Main execution
main() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    check_rosbag
    check_mesh
    check_mesh_directory
    make_executable
    
    while true; do
        show_menu
        read -p "Select option (1-8): " choice
        
        case $choice in
            1)
                play_rosbag_only
                ;;
            2)
                run_foundationpose_only
                ;;
            3)
                run_full_test
                ;;
            4)
                run_debug_mode
                ;;
            5)
                run_multi_object_test
                ;;
            6)
                check_rosbag_info
                ;;
            7)
                test_components
                ;;
            8)
                echo -e "${GREEN}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option. Please select 1-8.${NC}"
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
