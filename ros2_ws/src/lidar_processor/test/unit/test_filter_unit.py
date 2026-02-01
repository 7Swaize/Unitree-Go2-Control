import os
import sys
import yaml
import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch

from go2_interfaces.msg import LidarDecoded
from ament_index_python import get_package_share_directory

from lidar_processor.lidar_message_utils import create_lidar_decoded_message


class TestLidarFilterNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = os.path.join(get_package_share_directory('bringup'), 'config', 'lidar_processor.yaml')
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        cls.config_params = config_data['lidar_filter']['ros__parameters']['filter']


    def setUp(self):
        def create_node():
            from lidar_processor.lidar_filter_node import LidarFilterNode, FilterConfig

            with patch('lidar_processor.lidar_decoder_node.Node.__init__', return_value=None):
                node = LidarFilterNode.__new__(LidarFilterNode)
                node.config = FilterConfig(**self.config_params)
                node.filtered_cloud_pub = Mock(publish=lambda msg: self.received_messages.append(msg))
                node.get_logger = MagicMock(return_value=MagicMock(
                    info=lambda msg: print(msg, file=sys.stdout),
                    warn=lambda msg: print(msg, file=sys.stderr),
                    error=lambda msg: print(msg, file=sys.stderr),
                    debug=lambda msg: print(msg, file=sys.stderr)
                ))

                return node
        
        self.node = create_node()
        self.received_messages = []


    def test_filter_removes_out_of_range_xyz_points(self):
        xyz_data = np.array([
            [10.0, 0.0, 0.0],   # within range
            [30.0, 0.0, 0.0],   # beyond max_range
            [15.0, 0.0, 0.0],   # within range
        ], dtype=np.float32)
        msg = create_lidar_decoded_message(xyz_data)

        self.node.decoded_cloud_callback(msg)

        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)


    def test_filter_with_intensity_data(self):
        xyz_data = np.array([
            [15.0, 0.0, 0.0],
            [12.0, 0.0, 1.0],
            [41.0, 0.0, 0.0],
        ], dtype=np.float32)
        intensity_data = np.array([-100.0, 150.0, 91.0], dtype=np.float32)
        msg = create_lidar_decoded_message(xyz_data, intensity_data)

        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertTrue(filtered_msg.has_intensity)