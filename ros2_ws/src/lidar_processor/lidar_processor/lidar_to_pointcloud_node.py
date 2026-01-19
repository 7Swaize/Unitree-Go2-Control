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


POINTFIELD_TO_INTERNAL_CTYPE = {
    PointField.INT8: fp.PF_INT8,
    PointField.UINT8: fp.PF_UINT8,
    PointField.INT16: fp.PF_INT16,
    PointField.UINT16: fp.PF_UINT16,
    PointField.INT32: fp.PF_INT32,
    PointField.UINT32: fp.PF_UINT32,
    PointField.FLOAT32: fp.PF_FLOAT32,
    PointField.FLOAT64: fp.PF_FLOAT64
}

NP_DTYPE_TO_CODE = {
    np.int8: 1,
    np.uint8: 2,
    np.int16: 3,
    np.uint16: 4,
    np.int32: 5,
    np.uint32: 6,
    np.float32: 7,
    np.float64: 8
}

@dataclass
class PointCloudCollectionConfig:
    optimize_collection: bool


class LidarToPointCloudNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_to_pointcloud")

        self._declare_parameters()
        self._config = self._load_configuration()

        self._qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self._setup_publishers()
        self._setup_subscriptions()

    
    def _declare_parameters(self) -> None:
        self.declare_parameter("optimize_collection", False)


    def _load_configuration(self) -> PointCloudCollectionConfig:
        optimize_collection: bool = self.get_parameter("optimize_collection").get_parameter_value().bool_value

        return PointCloudCollectionConfig(
            optimize_collection=optimize_collection
        )


    def _setup_publishers(self) -> None:
        self._decoded_pointcloud_pub = self.create_publisher(
            LidarDecoded,
            "utlidar/lidar_decoded",
            self._qos_profile
        )


    def _setup_subscriptions(self) -> None:
        self.cloud_subscription = self.create_subscription(
            PointCloud2,
            "/utlidar/cloud",
            self._lidar_callback_optimized if self._config.optimize_collection else self._lidar_callback_unoptimized,
            self._qos_profile
        )


    def _lidar_callback_unoptimized(self, msg: PointCloud2) -> None:
        try:
            gen = point_cloud2.read_points(
                msg,
                field_names=["x", "y", "z"],
                skip_nans=True
            )

            xyz = np.array(list(gen), dtype=np.float64)

            self._publish_decoded_pointcloud(xyz, None)

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")
            

    def _lidar_callback_optimized(self, msg: PointCloud2) -> None:
        try:
            fields = {f.name: f for f in msg.fields}
            if not all(k in fields for k in ("x", "y", "z")):
                raise ValueError("PointCloud2 missing XYZ fields")

            dtype_xyz = fields["x"].datatype
            if any(fields[k].datatype != dtype_xyz for k in ("x", "y", "z")):
                raise TypeError("Mixed XYZ datatypes not supported")

            has_intensity = "intensity" in fields and POINTFIELD_TO_INTERNAL_CTYPE.get(fields["intensity"].datatype)

            xyz, intensity = fp.decode_xyz_intensity(
                data=msg.data,
                point_step=msg.point_step,
                ox=fields["x"].offset,
                oy=fields["y"].offset,
                oz=fields["z"].offset,
                oi=fields["intensity"].offset if has_intensity else -1,
                is_bigendian=msg.is_bigendian,
                dtype_xyz=POINTFIELD_TO_INTERNAL_CTYPE[dtype_xyz],
                dtype_intensity=POINTFIELD_TO_INTERNAL_CTYPE[fields["intensity"].datatype] if has_intensity else fp.PF_INT8,
                skip_nans=True
            )

            self._publish_decoded_pointcloud(xyz, intensity)
        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")


    def _publish_decoded_pointcloud(self, xyz: np.ndarray, intensity: Optional[np.ndarray]) -> None:
        try:
            msg = LidarDecoded()

            msg.xyz_shape = list(xyz.shape)
            msg.xyz_dtype = NP_DTYPE_TO_CODE[xyz.dtype]
            msg.xyz_data = xyz.tobytes(order='C')

            if intensity is not None:
                msg.has_intensity = True
                msg.intensity_shape = list(intensity.shape)
                msg.intensity_dtype = NP_DTYPE_TO_CODE[intensity.dtype]
                msg.intensity_data = intensity.tobytes(order='C')
            else:
                msg.has_intensity = False
                msg.intensity_shape = []
                msg.intensity_dtype = 0
                msg.intensity_data = []

            self._decoded_pointcloud_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing point cloud: {e}")

            

def main(args=None):
    rclpy.init(args=args)

    try:
        node = LidarToPointCloudNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running lidar processor: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()