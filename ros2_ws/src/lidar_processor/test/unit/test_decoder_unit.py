import os
import sys
import yaml
import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch

from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from go2_interfaces.msg import LidarDecoded
from ament_index_python import get_package_share_directory


class TestLidarDecoderNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = os.path.join(get_package_share_directory('bringup'), 'config', 'lidar_processor.yaml')
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        cls.config_params = config_data['lidar_decoder']['ros__parameters']['collection']


    def setUp(self):
        def create_node():
            from lidar_processor.lidar_decoder_node import LidarDecoderNode, CollectionConfig

            with patch('lidar_processor.lidar_decoder_node.Node.__init__', return_value=None):
                node = LidarDecoderNode.__new__(LidarDecoderNode)
                node.config = CollectionConfig(**self.config_params)
                node.decoded_pointcloud_pub = Mock(publish=lambda msg: self.received_messages.append(msg))
                node.pc_layout = None
                
                node.get_logger = MagicMock(return_value=MagicMock(
                    info=lambda msg: print(msg, file=sys.stdout),
                    warn=lambda msg: print(msg, file=sys.stderr),
                    error=lambda msg: print(msg, file=sys.stderr),
                    debug=lambda msg: print(msg, file=sys.stderr)
                ))

                return node
        
        self.node = create_node()
        self.received_messages = []


    def create_mock_pointcloud2(self, xyz_data: np.ndarray, intensity_data: np.ndarray = None) -> PointCloud2:
        header = Header(frame_id='lidar')
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        if intensity_data is not None:
            fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
            points = np.hstack([
                xyz_data.astype(np.float32),
                intensity_data.reshape(-1, 1).astype(np.float32)
            ])
        else:
            points = xyz_data.astype(np.float32)

        return point_cloud2.create_cloud(header, fields, points)

    
    def test_decoder_publishes_on_valid_xyz_pc(self):
        xyz_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        cloud = self.create_mock_pointcloud2(xyz_data)

        self.node.lidar_callback_optimized(cloud) if self.node.config.optimize_collection else self.node.lidar_callback_unoptimized(cloud)
        
        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertSequenceEqual(list(filtered_msg.xyz_shape), [3, 3])
        self.assertGreater(len(filtered_msg.xyz_data), 0)
        self.assertFalse(filtered_msg.has_intensity)
    

    def test_decoder_publishes_on_valid_xyzi_pc(self):
        xyz_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)
        intensity_data = np.array([100.0, 200.0], dtype=np.float32)
        cloud = self.create_mock_pointcloud2(xyz_data, intensity_data)

        self.node.lidar_callback_optimized(cloud) if self.node.config.optimize_collection else self.node.lidar_callback_unoptimized(cloud)

        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertSequenceEqual(list(filtered_msg.xyz_shape), [2, 3])
        self.assertGreater(len(filtered_msg.xyz_data), 0)  

        self.assertTrue(filtered_msg.has_intensity)
        self.assertSequenceEqual(list(filtered_msg.intensity_shape), [2])
        self.assertGreater(len(filtered_msg.xyz_data), 0)

    
    def test_decoder_handles_empty_pc(self):
        xyz_data = np.array([], dtype=np.float32).reshape(0, 3)
        intensity_data = np.array([], dtype=np.float32).reshape(0, 1)
        cloud = self.create_mock_pointcloud2(xyz_data, intensity_data)

        self.node.lidar_callback_optimized(cloud) if self.node.config.optimize_collection else self.node.lidar_callback_unoptimized(cloud)

        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertSequenceEqual(list(filtered_msg.xyz_shape), [0, 3])
        self.assertTrue(filtered_msg.has_intensity)
        self.assertSequenceEqual(list(filtered_msg.intensity_shape), [0])



'''
export ASAN_OPTIONS="detect_leaks=0:halt_on_error=1:symbolize=1"
export UBSAN_OPTIONS="print_stacktrace=1"
LD_PRELOAD=$(gcc -print-file-name=libasan.so) colcon test
'''

'''
LD_PRELOAD=$(gcc -print-file-name=libasan.so) \
pytest src/lidar_processor/test/unit/test_decoder_unit.py -s
'''