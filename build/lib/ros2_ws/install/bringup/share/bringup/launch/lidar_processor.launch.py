from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('bringup'), 'config')
    params_file = os.path.join(config_dir, 'lidar_processor.yaml')

    return LaunchDescription([
        Node(
            package='lidar_processor',
            executable='lidar_decoder_node',
            name='lidar_decoder',
            parameters=[params_file],
            output='screen'
        ),
        Node(
            package='lidar_processor',
            executable='lidar_filter_node',
            name='lidar_filter',
            parameters=[params_file],
            output='screen'
        ),
        Node(
            package='lidar_processor',
            executable='ros_bridge_node',
            name='ros_bridge',
            output='screen'
        )
    ])