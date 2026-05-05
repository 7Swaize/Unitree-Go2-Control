import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('bringup'), 'config')
    params_file = os.path.join(config_dir, 'lidar_processor.yaml')

    return LaunchDescription([
        Node(
            package='lidar_processor',
            executable='combined_lidar_node',
            parameters=[params_file],
            output='screen'
        )
    ])