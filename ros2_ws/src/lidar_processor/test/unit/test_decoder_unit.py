import time
import numpy as np
import unittest

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointField, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from go2_interfaces.msg import LidarDecoded


class TestLidarDecoderNode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    def setUp(self):
        self.node = rclpy.create_node('test_decoder_unit')
        self.received_messages = []

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.subscription = self.node.create_subscription(
            LidarDecoded,
            'utlidar/decoded_cloud',
            lambda msg: self.received_messages.append(msg),
            qos
        )

        self.publisher = self.node.create_publisher(
            PointCloud2,
            "/utlidar/cloud",
            qos
        )


    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def tearDown(self):
        self.node.destroy_subscription(self.subscription)
        self.node.destroy_publisher(self.publisher)
        self.node.destroy_node()


    def create_mock_pointcloud2(self, xyz_data: np.ndarray, intensity_data: np.ndarray = None):
        header = Header(frame_id='lidar')

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        if intensity_data is not None:
            fields.append(PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1))
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

        self.publisher.publish(cloud)

        end_time = time.time() + 5 
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.received_messages:
                break


        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertEqual(filtered_msg.xyz_shape, [3, 3])
        self.assertGreater(len(filtered_msg.xyz_data), 0)
        self.assertFalse(filtered_msg.has_intensity)
    

    def test_decoder_publishes_on_valid_xyzi_pc(self):
        xyz_data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)
        intensity_data = np.array([100.0, 200.0], dtype=np.float32)
        cloud = self.create_mock_pointcloud2(xyz_data, intensity_data)

        self.publisher.publish(cloud)

        end_time = time.time() + 5 
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.received_messages:
                break


        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertEqual(filtered_msg.xyz_shape, [2, 3])
        self.assertGreater(len(filtered_msg.xyz_data), 0)  

        self.assertTrue(filtered_msg.has_intensity)
        self.assertEqual(filtered_msg.intensity_shape, [2, 1])
        self.assertGreater(len(filtered_msg.xyz_data), 0)

    
    def test_decoder_handles_empty_pc(self):
        xyz_data = np.array([], dtype=np.float32).reshape(0, 3)
        intensity_data = np.array([], dtype=np.float32).reshape(0, 1)
        cloud = self.create_mock_pointcloud2(xyz_data, intensity_data)

        self.publisher.publish(cloud)

        end_time = time.time() + 5 
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.received_messages:
                break


        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertIsInstance(filtered_msg, LidarDecoded)
        self.assertEqual(filtered_msg.xyz_shape, [0, 3])
        self.assertTrue(filtered_msg.has_intensity)
        self.assertEqual(filtered_msg.intensity_shape, [0, 1])