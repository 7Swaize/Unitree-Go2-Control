import os

import pytest
from launch import LaunchDescription
import launch_testing.actions
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


@pytest.mark.launch_test
def generate_test_description():
    params_file = os.path.join(get_package_share_directory('bringup'), 'config', 'lidar_processor.yaml')

    decoder_node = Node(
        package='lidar_processor',
        executable='lidar_decoder_node',
        name='lidar_decoder',
        parameters=[params_file],
        additional_env={'PYTHONUNBUFFERED': '1'},
        output='screen'
    )

    filter_node = Node(
        package='lidar_processor',
        executable='lidar_filter_node',
        name='lidar_filter',
        parameters=[params_file],
        additional_env={'PYTHONUNBUFFERED': '1'},
        output='screen'
    )

    return LaunchDescription([
        decoder_node,
        filter_node,
        launch_testing.actions.ReadyToTest(),
    ]), {
        'lidar_decoder': decoder_node,
        'lidar_filter': filter_node
    }




'''
Command for running tests with cleaner logging:
colcon test \
  --packages-select lidar_processor \
  --ctest-args launch.py \
  --pytest-args -v --tb=short --disable-warnings --color=no
'''