import os
import sys
import yaml
import numpy as np
import unittest
from unittest.mock import Mock, MagicMock, patch

from std_msgs.msg import Header
from ament_index_python import get_package_share_directory

from lidar_processor.lidar_filter_node import LidarFilterNode, FilterConfig
from lidar_processor.lidar_message_utils import (
    create_lidar_decoded_message
)


class TestLidarFilterNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        yaml_path = os.path.join(get_package_share_directory('bringup'), 'config', 'lidar_processor.yaml')
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)

        cls.config_params = config_data['lidar_filter']['ros__parameters']['filter']


    def setUp(self):
        def create_node():
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


    def collect_ndarray(self, cloud_filtered: np.ndarray, src_pc_header: Header):
        xyz = cloud_filtered[:, :3]
        intensity = cloud_filtered[:, 3:] if cloud_filtered.shape[1] > 3 else None
        self.captured_xyz = xyz
        self.captured_intensity = intensity

    
    def validate_sor(self, outliers):
        sor_radius = self.node.config.sor_radius
        sor_min_neighbors = self.node.config.sor_min_neighbors
        voxel_indices = np.floor(self.captured_xyz / sor_radius).astype(int)

        voxels, counts = np.unique([tuple(v) for v in voxel_indices], return_counts=True, axis=0)

        for v, c in zip(voxels, counts):
            self.assertGreaterEqual(
                c,
                sor_min_neighbors,
                f"Voxel {v} has only {c} points, less than sor_min_neighbors {sor_min_neighbors}"
            )

        for outlier in outliers:
            dists = np.linalg.norm(self.captured_xyz - outlier, axis=1)
            self.assertTrue(np.all(dists > 1e-6), f"Outlier {outlier} not removed by SOR")


    def test_c_filter_result(self):
        num_points = 15000
        xyz_data = np.random.default_rng().uniform(low=-5, high=25, size=(num_points, 3))
        intensity_data = np.random.default_rng().uniform(low=-50, high=200, size=(num_points, 1))

        # filter outliers
        xyz_data[0] = [self.node.config.min_range - 1, 0, 0]
        xyz_data[1] = [self.node.config.max_range + 1, 0, 0] 
        xyz_data[2] = [10, self.node.config.height_max + 1, 0]
        intensity_data[3] = [self.node.config.intensity_min - 10]

        # sor outliers
        outliers = np.array([[50, 50, 50], [60, -40, 30], [55, 0, -45]])
        xyz_data = np.vstack([xyz_data, outliers])
        intensity_data = np.vstack([intensity_data, [[100], [150], [200]]])

        msg = create_lidar_decoded_message(xyz_data, intensity_data, Header())
        with patch.object(LidarFilterNode, 'publish_filtered_pointcloud', side_effect=self.collect_ndarray):
            self.node.decoded_cloud_callback(msg)

        self.assertTrue(np.all(np.linalg.norm(self.captured_xyz[:, :2], axis=1) >= self.node.config.min_range))
        self.assertTrue(np.all(np.linalg.norm(self.captured_xyz[:, :2], axis=1) <= self.node.config.max_range))
        self.assertTrue(np.all(self.captured_xyz[:, 2] >= self.node.config.height_min))
        self.assertTrue(np.all(self.captured_xyz[:, 2] <= self.node.config.height_max))
        self.assertTrue(np.all(self.captured_intensity >= self.node.config.intensity_min))

        self.assertEqual(self.captured_xyz.shape[0], self.captured_intensity.shape[0])
        self.assertEqual(self.captured_xyz.shape[1], 3)
        self.assertEqual(self.captured_intensity.shape[1], 1)

        self.validate_sor(outliers)
