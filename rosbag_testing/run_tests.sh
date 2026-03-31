#!/bin/bash

# ============================================================================
# RosBag FoundationPose Testing Script with Multi-Object Support
# ----------------------------------------------------------------------------
# Provides interactive test scenarios to:
#   * Replay a ROS 2 bag (inspection, debug, multi-object)
#   * Launch FoundationPose node with different modes (normal / debug / multi)
#   * Launch auxiliary TF static transforms and RViz visualization
#   * Run pre-flight diagnostics (mesh + rosbag availability, imports, etc.)
#   * Offer a menu-driven loop for repeated experimentation
#
# The script is intentionally verbose and defensive; each major step is
# commented so new users can quickly understand what happens under the hood.
# ============================================================================

set -e  # Exit immediately if any simple command returns non-zero (fail fast)

# ------------------------- Color escape sequences ---------------------------
# Used purely for nicer terminal output readability.
RED='\033[0;31m'      # Error / critical messages
GREEN='\033[0;32m'    # Success confirmations
YELLOW='\033[1;33m'   # Warnings / guidance
BLUE='\033[0;34m'     # Section headers / progress
NC='\033[0m'          # Reset / no color

# ----------------------------- Core path setup ------------------------------
# NOTE: Case sensitivity matters on Linux. Ensure this path matches the actual
# directory name. Adjust if your clone path differs.
FOUNDATIONPOSE_ROOT="/home/student/Desktop/Perception/FoundationPose"  # Root of project
ROSBAG_TESTING_DIR="${FOUNDATIONPOSE_ROOT}/rosbag_testing"              # This script's utilities
ROSBAG_PATH="${FOUNDATIONPOSE_ROOT}/demo_data/jonas_data"               # Folder containing the target rosbag
MESH_FILE="demo_data/cube/model_vhacd.obj"                              # Single mesh used for single-object tests
MESH_DIRECTORY="demo_data"                                             # Directory potentially containing many meshes for multi-object mode

echo -e "${BLUE}=== RosBag FoundationPose Testing Suite ===${NC}"
echo -e "${BLUE}Root Directory: ${FOUNDATIONPOSE_ROOT}${NC}"
echo -e "${BLUE}RosBag Path: ${ROSBAG_PATH}${NC}"
echo -e "${BLUE}Mesh File: ${MESH_FILE}${NC}"
echo -e "${BLUE}Mesh Directory: ${MESH_DIRECTORY}${NC}"

# ------------------------- Validation: rosbag files -------------------------
# Checks for presence of a specific rosbag .mcap file and its metadata. Adjust
# filename if using a different recording. Exits script if missing.
check_rosbag() {
    # Validate main bag file existence
    if [ ! -f "${ROSBAG_PATH}/rosbag2_2025_05_23-11_03_48_0.mcap" ]; then
        echo -e "${RED}Error: RosBag file not found at ${ROSBAG_PATH}/rosbag2_2025_05_23-11_03_48_0.mcap${NC}"  # Failure message
        exit 1  # Abort early—other actions depend on the bag
    fi

    # Validate ROS 2 metadata file presence (required for playback)
    if [ ! -f "${ROSBAG_PATH}/metadata.yaml" ]; then
        echo -e "${RED}Error: RosBag metadata not found at ${ROSBAG_PATH}/metadata.yaml${NC}"  # Failure message
        exit 1  # Abort early
    fi

    echo -e "${GREEN}✓ RosBag files found${NC}"  # Success confirmation
}

# --------------------------- Validation: mesh file --------------------------
# Ensures a reference mesh file exists, required for single-object pose runs.
check_mesh() {
    if [ ! -f "${FOUNDATIONPOSE_ROOT}/${MESH_FILE}" ]; then
        echo -e "${RED}Error: Mesh file not found at ${FOUNDATIONPOSE_ROOT}/${MESH_FILE}${NC}"  # Missing mesh
        exit 1  # Cannot proceed with pose estimation without mesh
    fi

    echo -e "${GREEN}✓ Mesh file found${NC}"  # Mesh ready
}

# ---------------------- Validation: mesh directory summary ------------------
# Ensures a directory of potential meshes exists for multi-object detection.
check_mesh_directory() {
    if [ ! -d "${FOUNDATIONPOSE_ROOT}/${MESH_DIRECTORY}" ]; then
        echo -e "${RED}Error: Mesh directory not found at ${FOUNDATIONPOSE_ROOT}/${MESH_DIRECTORY}${NC}"  # Missing dir
        exit 1
    fi

    # Count available mesh geometry files (OBJ / STL) for operator visibility
    mesh_count=$(find "${FOUNDATIONPOSE_ROOT}/${MESH_DIRECTORY}" -name "*.obj" -o -name "*.stl" -o -name "*.STL" | wc -l)  # Tally meshes
    echo -e "${GREEN}✓ Mesh directory found with ${mesh_count} mesh files${NC}"  # Report inventory
}

# ------------------------- Permissions: helper scripts ----------------------
# Ensures auxiliary scripts are executable (RViz launcher, system node if needed).
make_executable() {
    chmod +x "${ROSBAG_TESTING_DIR}/fixed_rviz.sh"        # Allow direct execution of RViz wrapper
    chmod +x "${ROSBAG_TESTING_DIR}/system_ros_node.py"   # Ensure Python node can be run (optional convenience)
    echo -e "${GREEN}✓ Scripts made executable${NC}"       # Confirmation
}

# ----------------------------- UI: menu printer -----------------------------
# Displays the interactive menu of available test scenarios.
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

# ------------------------- Option 1: rosbag playback ------------------------
# Plays the target bag at reduced speed for inspection. User can inspect topics
# or image streams in separate terminals / GUI tools.
play_rosbag_only() {
    echo -e "${BLUE}Playing RosBag only...${NC}"
    echo -e "${YELLOW}In another terminal, you can run: ros2 topic list${NC}"
    echo -e "${YELLOW}To see image topics: ros2 run rqt_image_view rqt_image_view${NC}"
    
    cd "${ROSBAG_PATH}"                 # Move into the bag directory (ros2 bag expects this)
    ros2 bag play . --loop --rate 0.5   # Replay continuously at half speed for easier observation
}

# -------------------- Option 2: FoundationPose node only --------------------
# Launches the Python node alone. Requires the user to start rosbag separately
# so node can subscribe to incoming image / camera topics.
run_foundationpose_only() {
    echo -e "${BLUE}Running FoundationPose node only...${NC}"
    echo -e "${YELLOW}Make sure to play the rosbag in another terminal first!${NC}"
    echo -e "${YELLOW}Command: cd ${ROSBAG_PATH} && ros2 bag play . --loop --rate 0.5${NC}"
    
    export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:$PYTHONPATH"  # Ensure project modules are discoverable

    cd "${ROSBAG_TESTING_DIR}"                              # Move to script directory for relative imports
    /usr/bin/python3 system_ros_node.py                      # Run node (foreground)
}

# -------------------- Option 3: Integrated full test run --------------------
# Orchestrates bag playback, required TF static transforms, FoundationPose,
# and RViz visualization. Manages lifetimes & cleanup via trap.
run_full_test() {
    echo -e "${BLUE}Running full integrated test...${NC}"
    
    export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:$PYTHONPATH"           # Make project importable

    echo -e "${BLUE}Starting RosBag player...${NC}"                 # Announce rosbag playback
    cd "${ROSBAG_PATH}"                                            # Enter rosbag directory
    ros2 bag play . --loop --rate 0.5 &                             # Start background playback
    ROSBAG_PID=$!                                                   # Capture process ID

    sleep 3  # Allow topics to initialize before launching downstream consumers

    echo -e "${BLUE}Starting transforms...${NC}"                    # Announce TF static publishers
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 \
        camera_link camera_color_frame &                           # Identity transform for base to color frame
    TF1_PID=$!                                                      # Capture PID
    ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 \
        camera_color_frame camera_color_optical_frame &            # Rotate to optical frame conventions
    TF2_PID=$!                                                      # Capture PID

    sleep 2  # Give TF a moment to broadcast

    echo -e "${BLUE}Starting FoundationPose...${NC}"                # Announce node start
    cd "${ROSBAG_TESTING_DIR}"                                      # Move into code directory
    /usr/bin/python3 system_ros_node.py &                           # Launch node (background)
    FP_PID=$!                                                       # Capture PID

    sleep 3  # Let node subscribe & warm up

    echo -e "${BLUE}Starting RViz...${NC}"                          # Announce visualization
    "${ROSBAG_TESTING_DIR}/fixed_rviz.sh" -d \
        "${ROSBAG_TESTING_DIR}/detection_visualization.rviz" &     # Launch RViz with predefined layout
    RVIZ_PID=$!                                                     # Capture PID

    echo ""                                                         # Blank line for readability
    echo -e "${GREEN}All components started!${NC}"                  # Success banner
    echo -e "${YELLOW}PIDs: RosBag=${ROSBAG_PID}, TF1=${TF1_PID}, TF2=${TF2_PID}, FoundationPose=${FP_PID}, RViz=${RVIZ_PID}${NC}"  # Process summary
    echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"      # User guidance

    trap 'echo -e "\n${YELLOW}Stopping all processes...${NC}"; \
          kill $ROSBAG_PID $TF1_PID $TF2_PID $FP_PID $RVIZ_PID 2>/dev/null; exit 0' INT  # Cleanup trap
    wait  # Block until a background job exits (Ctrl+C expected)
}

# ---------------------- Option 4: Debug (slower + verbose) ------------------
# Launches a slower rosbag playback and enables DEBUG logging for deeper issue
# investigation. Skips RViz to reduce noise unless user opens it manually.
run_debug_mode() {
    echo -e "${BLUE}Running debug mode...${NC}"
    
    export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:$PYTHONPATH"         # Ensure module resolution

    echo -e "${BLUE}Starting RosBag player...${NC}"               # Announce playback
    cd "${ROSBAG_PATH}"                                          # Enter bag location
    ros2 bag play . --loop --rate 0.2 &                          # Slow rate for step-by-step debugging
    ROSBAG_PID=$!                                                # Capture PID

    sleep 2  # Allow initial messages to appear

    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 \
        camera_link camera_color_frame &                       # First static transform
    TF1_PID=$!                                                  # PID store
    ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 \
        camera_color_frame camera_color_optical_frame &        # Optical frame transform
    TF2_PID=$!                                                  # PID store

    sleep 1  # Minor delay before node start

    echo -e "${BLUE}Starting FoundationPose in debug mode...${NC}"  # Announce node launch
    cd "${ROSBAG_TESTING_DIR}"                                    # Enter code dir
    export RCUTILS_LOGGING_SEVERITY=DEBUG                        # Increase ROS 2 logging verbosity
    /usr/bin/python3 -u system_ros_node.py &                     # -u for unbuffered stdout (live logs)
    FP_PID=$!                                                    # PID store

    echo ""                                                      # Blank line
    echo -e "${GREEN}Debug mode started!${NC}"                     # Success message
    echo -e "${YELLOW}PIDs: RosBag=${ROSBAG_PID}, TF1=${TF1_PID}, TF2=${TF2_PID}, FoundationPose=${FP_PID}${NC}"  # Process summary
    echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"      # User guidance

    trap 'echo -e "\n${YELLOW}Stopping all processes...${NC}"; \
          kill $ROSBAG_PID $TF1_PID $TF2_PID $FP_PID 2>/dev/null; exit 0' INT  # Cleanup trap
    wait  # Wait for interrupt
}

# --------------- Option 5: Multi-object detection integrated run ------------
# Similar to full test but launches FoundationPose with multi-object mode
# flags and slightly slower playback to accommodate heavier processing.
run_multi_object_test() {
    echo -e "${BLUE}Running multi-object detection test...${NC}"
    echo -e "${YELLOW}This test will detect multiple objects in the rosbag video${NC}"
    
    check_mesh_directory                                           # Verify multi-object mesh inventory

    export PYTHONPATH="${FOUNDATIONPOSE_ROOT}:$PYTHONPATH"         # Project import path

    echo -e "${BLUE}Starting RosBag player...${NC}"               # Start playback announcement
    cd "${ROSBAG_PATH}"                                          # Enter bag dir
    ros2 bag play . --loop --rate 0.3 &                           # Moderate speed to balance throughput
    ROSBAG_PID=$!                                                # PID store

    sleep 3  # Wait for initial bursts of messages

    echo -e "${BLUE}Starting transforms...${NC}"                  # TF startup
    ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 \
        camera_link camera_color_frame &                       # Base to color frame
    TF1_PID=$!                                                  # PID store
    ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 \
        camera_color_frame camera_color_optical_frame &        # Color to optical
    TF2_PID=$!                                                  # PID store

    sleep 2  # Let TF tree settle

    echo -e "${BLUE}Starting FoundationPose in multi-object mode...${NC}"  # Node start
    cd "${ROSBAG_TESTING_DIR}"                                    # Enter code dir

    /usr/bin/python3 system_ros_node.py \
        --multi_object_mode true \
        --mesh_directory "${MESH_DIRECTORY}" \
        --camera_frame camera_color_optical_frame \
        --object_frame multi_objects \
        --debug true \
        --track_refine_iter 2 \
        --est_refine_iter 4 &                                    # Launch w/ multi-object params
    FP_PID=$!                                                    # PID store

    sleep 3  # Allow model initialization

    echo -e "${BLUE}Starting RViz for multi-object visualization...${NC}"  # Visualization start
    "${ROSBAG_TESTING_DIR}/fixed_rviz.sh" -d \
        "${ROSBAG_TESTING_DIR}/detection_visualization.rviz" &   # Launch RViz
    RVIZ_PID=$!                                                   # PID store

    echo ""                                                      # Spacing
    echo -e "${GREEN}Multi-object detection started!${NC}"         # Success banner
    echo -e "${YELLOW}Expected behavior:${NC}"                    # User guidance
    echo -e "  • Multiple colored cubes will be detected automatically"  # Behavior note
    echo -e "  • Each object gets its own pose topic: /object_X_pose"     # Topic naming
    echo -e "  • Markers will be published to: /object_markers"          # Marker topic
    echo -e "  • Processing may be slower due to multiple objects"       # Performance caveat
    echo -e "${YELLOW}PIDs: RosBag=${ROSBAG_PID}, TF1=${TF1_PID}, TF2=${TF2_PID}, FoundationPose=${FP_PID}, RViz=${RVIZ_PID}${NC}"  # Process list
    echo -e "${YELLOW}Press Ctrl+C to stop all processes${NC}"      # Stop instructions

    trap 'echo -e "\n${YELLOW}Stopping all processes...${NC}"; \
          kill $ROSBAG_PID $TF1_PID $TF2_PID $FP_PID $RVIZ_PID 2>/dev/null; exit 0' INT  # Cleanup trap
    wait  # Block until user interrupts
}

# ------------------------- Option 6: rosbag metadata info -------------------
# Displays structural / topic info about the current bag.
check_rosbag_info() {
    echo -e "${BLUE}Checking RosBag information...${NC}"
    cd "${ROSBAG_PATH}"     # Enter bag directory
    ros2 bag info .          # Query bag metadata
}

# --------------------- Option 7: component self-tests ----------------------
# Performs a sequence of lightweight checks without launching full pipeline.
test_components() {
    echo -e "${BLUE}Testing individual components...${NC}"
    
    echo -e "${YELLOW}1. Testing system_ros_node.py import...${NC}"
    cd "${ROSBAG_TESTING_DIR}"   # Move to module location for relative import context
    /usr/bin/python3 -c "        # Inline Python to test importability
import sys
sys.path.insert(0, '${FOUNDATIONPOSE_ROOT}')
try:
    from system_ros_node import SystemFoundationPoseNode
    print('✓ system_ros_node.py imports successfully')
except Exception as e:
    print(f'✗ Import failed: {e}')
"
    
    echo ""
    echo -e "${YELLOW}2. Testing RViz script...${NC}"
    if [ -x "${ROSBAG_TESTING_DIR}/fixed_rviz.sh" ]; then
        echo "✓ fixed_rviz.sh is executable"
    else
        echo "✗ fixed_rviz.sh is not executable"
    fi
    
    echo ""
    echo -e "${YELLOW}3. Testing RosBag access...${NC}"
    check_rosbag
    
    echo ""
    echo -e "${YELLOW}4. Testing mesh files...${NC}"
    check_mesh
    check_mesh_directory
}

# ------------------------------ Main dispatcher -----------------------------
# Runs prerequisite validations once, then enters a persistent menu loop.
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

# ---------------------------- Entry point guard -----------------------------
# Ensures main() only runs when this file is executed directly, not sourced.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
