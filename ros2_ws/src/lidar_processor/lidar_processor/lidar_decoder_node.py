from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from go2_interfaces.msg import LidarDecoded

import numpy as np
import fast_pointcloud as fp

from lidar_processor.lidar_message_utils import (
    POINTFIELD_TO_INTERNAL_CTYPE,
    create_lidar_decoded_message
)


@dataclass
class CollectionConfig:
    optimize_collection: bool
    skip_nans: bool 


@dataclass(frozen=True)
class PointCloudLayout:
    has_intensity: bool
    x_offset: int
    y_offset: int
    z_offset: int
    intensity_offset: int
    xyz_internal_type: int
    intensity_internal_type: int


class LidarDecoderNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_decoder")
        self._declare_parameters()
        self.config = self._load_configuration()

        self._qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.pc_layout: Optional[PointCloudLayout] = None

        self.setup_publishers()
        self.setup_subscriptions()

    
    def _declare_parameters(self) -> None:
        self.declare_parameters(
            namespace="",
            parameters=[
                ("collection.optimize_collection", rclpy.Parameter.Type.BOOL),
                ("collection.skip_nans", rclpy.Parameter.Type.BOOL),   
            ]
        )


    def _load_configuration(self) -> CollectionConfig:
        optimize_collection = self.get_parameter('collection.optimize_collection').get_parameter_value().bool_value
        skip_nans = self.get_parameter('collection.skip_nans').get_parameter_value().bool_value

        return CollectionConfig(
            optimize_collection=optimize_collection,
            skip_nans=skip_nans,
        )


    def setup_publishers(self) -> None:
        self.decoded_pointcloud_pub = self.create_publisher(
            LidarDecoded,
            "utlidar/decoded_cloud",
            self._qos_profile
        )


    def setup_subscriptions(self) -> None:
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            "/utlidar/cloud",
            self.lidar_callback_optimized if self.config.optimize_collection else self.lidar_callback_unoptimized,
            self._qos_profile
        )


    def _init_pointcloud_layout(self, msg: PointCloud2) -> PointCloudLayout:
        fields: dict[str, PointField] = {f.name: f for f in msg.fields}
        if not all(k in fields for k in ("x", "y", "z")):
            raise ValueError("PointCloud2 missing XYZ fields")

        dtype_xyz = fields["x"].datatype
        if not all(fields[k].datatype == dtype_xyz for k in ("x", "y", "z")):
            raise TypeError("Mixed XYZ datatypes not supported")
        
        xyz_internal = POINTFIELD_TO_INTERNAL_CTYPE.get(dtype_xyz)
        if xyz_internal is None:
            raise TypeError(f"Unsupported XYZ datatype: {dtype_xyz}")

        has_intensity = "intensity" in fields
        if has_intensity:
            intensity_dtype = fields["intensity"].datatype
            intensity_internal = POINTFIELD_TO_INTERNAL_CTYPE.get(intensity_dtype)
            if intensity_internal is None:
                raise TypeError(f"Unsupported intensity datatype: {intensity_dtype}")
            intensity_offset = fields["intensity"].offset
        else:
            intensity_internal = fp.PointFieldType["INT8"]
            intensity_offset = -1

        return PointCloudLayout(
            has_intensity=has_intensity,
            x_offset=fields["x"].offset,
            y_offset=fields["y"].offset,
            z_offset=fields["z"].offset,
            intensity_offset=intensity_offset,
            xyz_internal_type=xyz_internal,
            intensity_internal_type=intensity_internal
        )


    def lidar_callback_unoptimized(self, msg: PointCloud2) -> None:
        try:
            if self.pc_layout is None:
                self.pc_layout = self._init_pointcloud_layout(msg)

            if self.pc_layout.has_intensity:
                data = point_cloud2.read_points_numpy(
                    msg,
                    field_names=["x", "y", "z", "intensity"],
                    skip_nans=self.config.skip_nans
                ).astype(np.float32)

                xyz = data[:, :3]
                intensity = data[:, 3]
            else:
                xyz = point_cloud2.read_points_numpy(
                    msg,
                    field_names=["x", "y", "z"],
                    skip_nans=self.config.skip_nans
                ).astype(np.float32)
                intensity = None

            self.publish_decoded_pointcloud(xyz, intensity)

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")
            

    def lidar_callback_optimized(self, msg: PointCloud2) -> None:       
        try:
            if self.pc_layout is None:
                self.pc_layout = self._init_pointcloud_layout(msg)

            l = self.pc_layout

            xyz, intensity = fp.decode_xyz_intensity(
                msg.data,
                msg.point_step,
                l.x_offset,
                l.y_offset,
                l.z_offset,
                l.intensity_offset,
                msg.is_bigendian,
                l.xyz_internal_type,
                l.intensity_internal_type,
                self.config.skip_nans
            )

            self.publish_decoded_pointcloud(xyz, intensity)

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")


    def publish_decoded_pointcloud(self, xyz: np.ndarray, intensity: Optional[np.ndarray]) -> None:
        try:
            msg = create_lidar_decoded_message(xyz, intensity)
            self.decoded_pointcloud_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing point cloud: {e}")
            

def main(args=None):
    rclpy.init(args=args)

    try:
        node = LidarDecoderNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running lidar processor: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()