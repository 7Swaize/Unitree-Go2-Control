from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from go2_interfaces.msg import LidarDecoded

import numpy as np


@dataclass
class FilterConfig:
    max_range: float
    min_range: float
    height_min: float
    height_max: float
    downsample_rate: int
    sor_radius: float
    sor_min_neighbors: int
    intensity_min: float

    


class LidarFilterNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_filter")

        self.declare_parameters()
        self.config = self.load_configuration()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.setup_subscriptions()

    
    def declare_parameters(self):
        self.declare_parameter('max_range', 20.0)
        self.declare_parameter('min_range', 0.1)
        self.declare_parameter('height_filter_min', -2.0)
        self.declare_parameter('height_filter_max', 3.0)
        self.declare_parameter('downsample_rate', 10)
        self.declare_parameter('sor_radius', 0.5)
        self.declare_parameter('sor_min_neighbors', 3)
        self.declare_parameter('intensity_min', 0.0)
        self.declare_parameter('publish_rate', 5.0)


    def load_configuration(self) -> FilterConfig:
        return FilterConfig(
            max_range=self.get_parameter('max_range').get_parameter_value().double_value,
            min_range=self.get_parameter('min_range').get_parameter_value().double_value,
            height_min=self.get_parameter('height_filter_min').get_parameter_value().double_value,
            height_max=self.get_parameter('height_filter_max').get_parameter_value().double_value,
            downsample_rate=self.get_parameter('downsample_rate').get_parameter_value().integer_value,
            sor_radius=self.get_parameter('sor_radius').get_parameter_value().double_value,
            sor_min_neighbors=self.get_parameter('sor_min_neighbors').get_parameter_value().integer_value,
            intensity_min=self.get_parameter('intensity_min').get_parameter_value().double_value,
        )
    

    def setup_publishers(self):
        self.filtered_cloud_pub = self.create_publisher(
            LidarDecoded,
            "utlidar/filtered_cloud",
            self.qos_profile
        )
    

    def setup_subscriptions(self):
        self.decoded_cloud_subscription = self.create_subscription(
            LidarDecoded,
            "utlidar/decoded_cloud",
            self.decoded_cloud_callback,
            self.qos_profile
        )


    def decoded_cloud_callback(self, msg: LidarDecoded):
        pass
    
        # numpy from buffer and reshape
        # filter
        # pub