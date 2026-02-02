from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from go2_interfaces.msg import LidarDecoded
from std_msgs.msg import Header

import numpy as np
import fast_pointcloud as fp

from lidar_processor.lidar_message_utils import (
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
        self.config = self._load_configuration()

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.setup_publishers()
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

            cloud_filtered = fp.apply_filter(cloud_decoded, self.config)
            self.publish_filtered_pointcloud(cloud_filtered, msg.header)

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")
    

    def publish_filtered_pointcloud(self, cloud_filtered: np.ndarray, src_pc_header: Header) -> None:
        try:
            xyz = cloud_filtered[:, :3]
            intensity = cloud_filtered[:, 3:] if cloud_filtered.shape[1] > 3 else None
            
            msg = create_lidar_decoded_message(xyz, intensity.squeeze() if intensity is not None else None, src_pc_header)
            self.filtered_cloud_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing filtered point cloud: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = LidarFilterNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running lidar processor: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()