#!/usr/bin/env python3

"""
Launch file for RosBag FoundationPose testing with RViz visualization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    rosbag_path_arg = DeclareLaunchArgument(
        'rosbag_path',
        default_value='/home/student/Desktop/Perception/FoundationPose/demo_data/jonas_data/',
        description='Path to the RosBag directory'
    )
    
    mesh_file_arg = DeclareLaunchArgument(
        'mesh_file',
        default_value='demo_data/cube/model_vhacd.obj',
        description='Path to the mesh file relative to FoundationPose root'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Enable debug mode'
    )
    
    # Get launch configurations
    rosbag_path = LaunchConfiguration('rosbag_path')
    mesh_file = LaunchConfiguration('mesh_file')
    debug = LaunchConfiguration('debug')
    
    # RosBag player node
    rosbag_player = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop', '--rate', '0.5'],
        name='rosbag_player',
        output='screen'
    )
    
    # FoundationPose node
    foundationpose_node = Node(
        package='rosbag_testing',
        executable='rosbag_foundationpose_node.py',
        name='rosbag_foundationpose_node',
        output='screen',
        parameters=[{
            'foundationpose_path': '/home/student/Desktop/Perception/FoundationPose',
            'mesh_file': mesh_file,
            'camera_frame': 'camera_color_optical_frame',
            'object_frame': 'cube',
            'publish_rate': 10.0,
            'debug': debug,
            'confidence_threshold': 0.5
        }],
        remappings=[
            ('/camera/camera/color/image_raw', '/camera/camera/color/image_raw'),
            ('/camera/camera/depth/image_rect_raw', '/camera/camera/depth/image_rect_raw'),
            ('/camera/camera/color/camera_info', '/camera/camera/color/camera_info'),
        ]
    )
    
    # RViz node with delay to allow other nodes to start
    rviz_node = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', '/home/student/Desktop/Perception/FoundationPose/rosbag_testing/cube_detection.rviz'],
                output='screen'
            )
        ]
    )
    
    # Static transform publishers for camera frames
    camera_base_to_color_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'camera_link', 'camera_color_frame']
    )
    
    camera_color_to_optical_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '-1.570796', '0', '-1.570796', 'camera_color_frame', 'camera_color_optical_frame']
    )
    
    return LaunchDescription([
        rosbag_path_arg,
        mesh_file_arg,
        debug_arg,
        camera_base_to_color_tf,
        camera_color_to_optical_tf,
        rosbag_player,
        foundationpose_node,
        rviz_node,
    ])
