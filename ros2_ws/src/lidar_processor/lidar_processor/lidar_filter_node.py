import numpy as np
import fast_pointcloud as fp
from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from go2_interfaces.msg import LidarDecoded
from std_msgs.msg import Header

from .lidar_bridge import LidarBridge
from .lidar_message_utils import (
    decode_array_from_message,
    create_lidar_decoded_message
)


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

        self._declare_parameters()
        self._config = self._load_configuration()

        self._qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self._bridge = LidarBridge()
        self.setup_subscriptions()

    
    def _declare_parameters(self):
        self.declare_parameters(
            namespace="",
            parameters=[
                ("filter.max_range", rclpy.Parameter.Type.DOUBLE),
                ("filter.min_range", rclpy.Parameter.Type.DOUBLE),
                ("filter.height_min", rclpy.Parameter.Type.DOUBLE),
                ("filter.height_max", rclpy.Parameter.Type.DOUBLE),
                ("filter.downsample_rate", rclpy.Parameter.Type.INTEGER),
                ("filter.sor_radius", rclpy.Parameter.Type.DOUBLE),
                ("filter.sor_min_neighbors", rclpy.Parameter.Type.INTEGER),
                ("filter.intensity_min", rclpy.Parameter.Type.DOUBLE)
            ]
        )


    def _load_configuration(self) -> FilterConfig:
        max_range = self.get_parameter('filter.max_range').get_parameter_value().double_value
        min_range = self.get_parameter('filter.min_range').get_parameter_value().double_value
        height_min = self.get_parameter('filter.height_min').get_parameter_value().double_value
        height_max = self.get_parameter('filter.height_max').get_parameter_value().double_value
        downsample_rate = self.get_parameter('filter.downsample_rate').get_parameter_value().integer_value
        sor_radius = self.get_parameter('filter.sor_radius').get_parameter_value().double_value
        sor_min_neighbors = self.get_parameter('filter.sor_min_neighbors').get_parameter_value().integer_value
        intensity_min = self.get_parameter('filter.intensity_min').get_parameter_value().double_value

        return FilterConfig(
            max_range=max_range,
            min_range=min_range,
            height_min=height_min,
            height_max=height_max,
            downsample_rate=downsample_rate,
            sor_radius=sor_radius,
            sor_min_neighbors=sor_min_neighbors,
            intensity_min=intensity_min
        )
    

    def setup_subscriptions(self):
        self._decoded_cloud_subscription = self.create_subscription(
            LidarDecoded,
            "custom/decoded_cloud",
            self.decoded_cloud_callback,
            self._qos_profile
        )


    def decoded_cloud_callback(self, msg: LidarDecoded) -> None:
        try:
            xyz_decoded = decode_array_from_message(msg, "xyz")
            
            if msg.has_intensity:
                inten_decoded = decode_array_from_message(msg, "intensity")
                if inten_decoded.ndim == 1:
                    inten_decoded = inten_decoded[:, np.newaxis] # Makes sure inten is 2D
            
                cloud_decoded = np.hstack([xyz_decoded, inten_decoded])
            else:
                cloud_decoded = xyz_decoded

            cloud_filtered = fp.apply_filter(cloud_decoded, self._config)
            self.send_to_bridge(cloud_filtered, msg.header)

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")
    

    def send_to_bridge(self, cloud_filtered: np.ndarray, src_pc_header: Header) -> None:
        stamp_ns = src_pc_header.stamp.sec * 1_000_000_000 + src_pc_header.stamp.nanosec
        self._bridge.send_filtered(stamp_ns, cloud_filtered)