# ROS2 Jazzy - FoundationPose Integration

This integration allows you to use your existing FoundationPose setup with ROS2 Jazzy without Docker.

## Overview

The integration consists of:
- **ros2_bridge.py**: Main ROS2 node that interfaces with FoundationPose
- **test_publisher.py**: Test script to verify the integration
- **setup.sh**: Setup script for installing required packages

## Architecture

```
ROS2 Jazzy Environment    ←→    FoundationPose Environment
     (Python 3.12)               (Python 3.9 + venv)
         │                              │
    ros2_bridge.py    ←—————→    subprocess call
         │                              │
    ROS2 Messages                 File communication
```

## Setup

1. **Run the setup script** (already done):
   ```bash
   cd /home/student/Desktop/Perception/FoundationPose/ros2_integration
   ./setup.sh
   ```

## Usage

### Method 1: With Real Camera Data

1. **Start the ROS2 bridge** in one terminal:
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd /home/student/Desktop/Perception/FoundationPose/ros2_integration
   python3 ros2_bridge.py
   ```

2. **Start your camera node** in another terminal (example for RealSense):
   ```bash
   source /opt/ros/jazzy/setup.bash
   ros2 launch realsense2_camera rs_launch.py
   ```

### Method 2: With Test Data

1. **Start the ROS2 bridge** in one terminal:
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd /home/student/Desktop/Perception/FoundationPose/ros2_integration
   python3 ros2_bridge.py
   ```

2. **Run the test publisher** in another terminal:
   ```bash
   source /opt/ros/jazzy/setup.bash
   cd /home/student/Desktop/Perception/FoundationPose/ros2_integration
   python3 test_publisher.py
   ```

## Topics

### Subscribed Topics:
- `/camera/color/image_raw` (sensor_msgs/Image): RGB camera image
- `/camera/depth/image_raw` (sensor_msgs/Image): Depth camera image  
- `/camera/color/camera_info` (sensor_msgs/CameraInfo): Camera calibration

### Published Topics:
- `/foundationpose/pose` (geometry_msgs/PoseStamped): Estimated object pose

## Configuration

You can modify the parameters in `ros2_bridge.py`:

```python
# Parameters
self.declare_parameter('foundationpose_path', '/home/student/Desktop/Perception/FoundationPose')
self.declare_parameter('mesh_file', 'demo_data/mustard0/mesh/textured_simple.obj')
```

### Changing the Object

To detect a different object:

1. Place your object's `.obj` mesh file in the FoundationPose directory
2. Update the `mesh_file` parameter in `ros2_bridge.py`:
   ```python
   self.declare_parameter('mesh_file', 'path/to/your/object.obj')
   ```

## Monitoring

### View pose data:
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /foundationpose/pose
```

### View camera topics:
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep camera
```

### Check topic info:
```bash
source /opt/ros/jazzy/setup.bash
ros2 topic info /foundationpose/pose
```

## Troubleshooting

### Common Issues:

1. **"No module named 'rclpy'"**
   - Make sure you've sourced ROS2: `source /opt/ros/jazzy/setup.bash`

2. **FoundationPose estimation fails**
   - Check that your FoundationPose virtual environment is working
   - Verify the mesh file path is correct
   - Check camera calibration

3. **No camera data received**
   - Verify camera topics: `ros2 topic list | grep camera`
   - Check topic types: `ros2 topic type /camera/color/image_raw`

### Debug Mode:

Enable debug output by modifying the bridge script:
```python
# In ros2_bridge.py, set debug logging
self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
```

## Performance

- Processing rate: ~2 Hz (adjustable in timer)
- GPU memory: ~2GB required for FoundationPose
- CPU usage: Moderate (subprocess communication overhead)

## Files Structure

```
ros2_integration/
├── ros2_bridge.py          # Main ROS2 bridge node
├── test_publisher.py       # Test data publisher
├── setup.sh               # Setup script
├── foundationpose_node.py  # Alternative full ROS2 node
├── foundationpose_launch.py # Launch file
└── README.md              # This file
```

## Advanced Usage

### Integration with Navigation Stack

The pose output can be used with ROS2 navigation:

```bash
# Transform pose to map frame
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map camera_link

# Use with move_base or nav2
```

### Recording Data

```bash
# Record bag file
ros2 bag record /camera/color/image_raw /camera/depth/image_raw /foundationpose/pose

# Play back
ros2 bag play <bag_file>
```

## Next Steps

1. **Add segmentation**: Replace the simple mask with proper object segmentation
2. **Tracking mode**: Implement tracking for better performance
3. **Multiple objects**: Extend to detect multiple objects
4. **Calibration**: Add automatic camera-robot calibration
5. **Visualization**: Add RViz plugins for 3D visualization

## Support

For issues related to:
- FoundationPose: Check the original repository
- ROS2 integration: Check this README and troubleshooting section
- Camera drivers: Check respective camera ROS2 packages
