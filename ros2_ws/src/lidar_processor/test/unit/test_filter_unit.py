import time
import numpy as np
import unittest

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from go2_interfaces.msg import LidarDecoded

from lidar_processor.lidar_message_utils import create_lidar_decoded_message


'''
class TestLidarFilterNode(unittest.TestCase):
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
            'utlidar/filtered_cloud',
            lambda msg: self.received_messages.append(msg),
            qos
        )

        self.publisher = self.node.create_publisher(
            LidarDecoded,
            "/utlidar/decoded_cloud",
            qos
        )


    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def tearDown(self):
        self.node.destroy_subscription(self.subscription)
        self.node.destroy_publisher(self.publisher)
        self.node.destroy_node()


    def create_decoded_message(self, xyz: np.ndarray, intensity: np.ndarray = None):
        return create_lidar_decoded_message(xyz, intensity)
    

    def test_filter_removes_out_of_range_xyz_points(self):
        xyz_data = np.array([
            [10.0, 0.0, 0.0],   # within range
            [30.0, 0.0, 0.0],   # beyond max_range
            [15.0, 0.0, 0.0],   # within range
        ], dtype=np.float32)
        msg = self.create_decoded_message(xyz_data)

        self.publisher.publish(msg)
        
        end_time = time.time() + 5 
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.received_messages:
                break

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
        msg = self.create_decoded_message(xyz_data, intensity_data)

        self.publisher.publish(msg)

        end_time = time.time() + 5 
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.received_messages:
                break

        self.assertGreater(len(self.received_messages), 0)
        filtered_msg: LidarDecoded = self.received_messages[0]

        self.assertTrue(filtered_msg.has_intensity)



'''