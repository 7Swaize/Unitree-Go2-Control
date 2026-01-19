from dataclasses import dataclass

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarToPointCloudNode(Node):
    def __init__(self):
        super().__init__("lidar_to_pointcloud")

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.setup_subscriptions()
        self._setup_publishers()


    def setup_subscriptions(self):
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            "/utlidar/cloud",
            self._lidar_callback,
            self.qos_profile
        )

    
    def _lidar_callback(self, msg: PointCloud2):
        try:
            points = []
            


def main(args=None):
    pass


if __name__ == "__main__":
    main()