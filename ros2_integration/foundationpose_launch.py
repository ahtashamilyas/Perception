#!/usr/bin/env python3

"""
Launch file for FoundationPose ROS2 integration
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Declare launch arguments
    foundationpose_path_arg = DeclareLaunchArgument(
        'foundationpose_path',
        default_value='/home/student/Desktop/Perception/FoundationPose',
        description='Path to FoundationPose installation'
    )
    
    mesh_file_arg = DeclareLaunchArgument(
        'mesh_file',
        default_value='demo_data/mustard0/mesh/textured_simple.obj',
        description='Relative path to object mesh file'
    )
    
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_link',
        description='Camera frame ID'
    )
    
    object_frame_arg = DeclareLaunchArgument(
        'object_frame',
        default_value='object',
        description='Object frame ID'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Publishing rate in Hz'
    )
    
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='true',
        description='Enable debug output'
    )
    
    # FoundationPose node
    foundationpose_node = Node(
        package='foundationpose_ros2',
        executable='foundationpose_node',
        name='foundationpose_node',
        parameters=[{
            'foundationpose_path': LaunchConfiguration('foundationpose_path'),
            'mesh_file': LaunchConfiguration('mesh_file'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'object_frame': LaunchConfiguration('object_frame'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'debug': LaunchConfiguration('debug')
        }],
        output='screen'
    )
    
    return LaunchDescription([
        foundationpose_path_arg,
        mesh_file_arg,
        camera_frame_arg,
        object_frame_arg,
        publish_rate_arg,
        debug_arg,
        foundationpose_node
    ])
