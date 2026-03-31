# RosBag FoundationPose Testing

This package provides a complete solution for testing FoundationPose with recorded RosBag data containing cube objects. The system processes the RosBag data in real-time, detects cubes using FoundationPose, and visualizes the results in RViz.

## Overview

The testing suite includes:
- RosBag player for recorded camera data
- FoundationPose node for cube detection and pose estimation
- RViz configuration for visualization
- TF transforms for proper coordinate frame management
- Debug and monitoring tools

## Files Structure

```
rosbag_testing/
├── rosbag_foundationpose_node.py    # Main FoundationPose ROS2 node
├── cube_detection.rviz              # RViz configuration file
├── run_tests.sh                     # Comprehensive testing script
├── launch/
│   └── simple_launch.py             # Simple launch script
├── package.xml                      # ROS2 package definition
├── CMakeLists.txt                   # Build configuration
└── README.md                        # This file
```

## Prerequisites

1. **ROS2 Jazzy** properly installed and sourced
2. **FoundationPose** dependencies installed:
   - numpy
   - opencv-python
   - trimesh
   - torch (if using neural components)
3. **ROS2 packages**:
   - sensor_msgs
   - geometry_msgs
   - visualization_msgs
   - cv_bridge
   - tf2_ros
   - rviz2

## Data Requirements

- **RosBag file**: `/home/student/Desktop/Perception/FoundationPose/demo_data/jonas_data/rosbag2_2025_05_23-11_03_48_0.mcap`
- **Metadata**: `/home/student/Desktop/Perception/FoundationPose/demo_data/jonas_data/metadata.yaml`
- **Cube mesh**: `/home/student/Desktop/Perception/FoundationPose/demo_data/cube/model_vhacd.obj`

Required topics in RosBag:
- `/camera/camera/color/image_raw` (sensor_msgs/Image)
- `/camera/camera/depth/image_rect_raw` (sensor_msgs/Image)
- `/camera/camera/color/camera_info` (sensor_msgs/CameraInfo)

## Quick Start

### Method 1: Using the Comprehensive Testing Script

```bash
cd /home/student/Desktop/Perception/FoundationPose/rosbag_testing
chmod +x run_tests.sh
./run_tests.sh
```

Select option **3** for "Full integrated test with RViz"

### Method 2: Using Simple Launch Script

```bash
cd /home/student/Desktop/Perception/FoundationPose/rosbag_testing
chmod +x launch/simple_launch.py
python3 launch/simple_launch.py
```

### Method 3: Manual Step-by-Step

1. **Terminal 1 - Start RosBag**:
   ```bash
   cd /home/student/Desktop/Perception/FoundationPose/demo_data/jonas_data
   ros2 bag play . --loop --rate 0.5
   ```

2. **Terminal 2 - Start Static Transforms**:
   ```bash
   # Camera base to color frame
   ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 camera_link camera_color_frame
   ```

3. **Terminal 3 - Another Static Transform**:
   ```bash
   # Color frame to optical frame
   ros2 run tf2_ros static_transform_publisher 0 0 0 -1.570796 0 -1.570796 camera_color_frame camera_color_optical_frame
   ```

4. **Terminal 4 - Start FoundationPose Node**:
   ```bash
   cd /home/student/Desktop/Perception/FoundationPose/rosbag_testing
   python3 rosbag_foundationpose_node.py
   ```

5. **Terminal 5 - Start RViz**:
   ```bash
   rviz2 -d /home/student/Desktop/Perception/FoundationPose/rosbag_testing/cube_detection.rviz
   ```

## Expected Behavior

1. **RosBag Player**: Loops through recorded camera data at 0.5x speed
2. **FoundationPose Node**: 
   - Subscribes to camera topics
   - Processes images to detect cubes
   - Publishes pose estimations
   - Publishes visualization markers
   - Broadcasts TF transforms
3. **RViz Visualization**:
   - Shows live camera feed
   - Displays detected cube poses as red cubes
   - Shows coordinate frames for detected objects
   - Provides debug image with overlay

## Topics Published

- `/cube_pose` (geometry_msgs/PoseStamped) - Estimated cube poses
- `/cube_markers` (visualization_msgs/MarkerArray) - Visualization markers
- `/debug_image` (sensor_msgs/Image) - Debug image with overlays
- TF transforms: `camera_color_optical_frame` → `cube_0`, `cube_1`, etc.

## Configuration Parameters

The FoundationPose node accepts these parameters:

- `foundationpose_path`: Path to FoundationPose root directory
- `mesh_file`: Relative path to cube mesh file
- `camera_frame`: Camera optical frame name
- `object_frame`: Base name for detected object frames
- `publish_rate`: Rate for pose estimation (Hz)
- `debug`: Enable debug mode and debug image publishing
- `confidence_threshold`: Minimum confidence for detections

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure Python path includes FoundationPose
   export PYTHONPATH="/home/student/Desktop/Perception/FoundationPose:$PYTHONPATH"
   ```

2. **No Detections**:
   - Check if cube is visible in camera feed
   - Verify mesh file path is correct
   - Adjust confidence threshold
   - Check camera calibration

3. **TF Errors**:
   - Ensure static transform publishers are running
   - Check frame names match in RViz

4. **Performance Issues**:
   - Reduce RosBag playback rate
   - Lower publish_rate parameter
   - Disable debug mode

### Debug Mode

Run with verbose logging:
```bash
export RCUTILS_LOGGING_SEVERITY=DEBUG
python3 rosbag_foundationpose_node.py
```

### Monitoring Tools

Check topics:
```bash
ros2 topic list
ros2 topic echo /cube_pose
ros2 topic hz /camera/camera/color/image_raw
```

View images:
```bash
ros2 run rqt_image_view rqt_image_view
```

Check TF tree:
```bash
ros2 run tf2_tools view_frames
```

## Testing Options

The `run_tests.sh` script provides several testing modes:

1. **Play RosBag only**: For topic inspection
2. **Run FoundationPose only**: Manual rosbag control
3. **Full integrated test**: Complete automated setup
4. **Debug mode**: Verbose output and slower playback
5. **Check RosBag info**: Inspect bag contents
6. **Test components**: Individual component testing

## Expected Results

- **Live Detection**: Cubes should be detected and tracked in real-time
- **Pose Accuracy**: 6DOF poses should be reasonable for visible cubes
- **Visualization**: Red cube markers should overlay detected objects
- **Stability**: Poses should be stable with minimal jitter
- **Performance**: System should run at ~10 Hz on modern hardware

## Customization

To use different objects:
1. Replace mesh file path in parameters
2. Adjust marker scale in visualization
3. Modify confidence thresholds as needed
4. Update RViz configuration for different visualization

## Support

For issues specific to:
- **FoundationPose**: Check original FoundationPose documentation
- **ROS2 Integration**: Verify ROS2 Jazzy installation
- **Camera Topics**: Ensure RealSense data is properly recorded
- **Visualization**: Check RViz configuration and TF tree
