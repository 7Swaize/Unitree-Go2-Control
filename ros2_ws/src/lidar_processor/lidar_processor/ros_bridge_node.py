import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from go2_interfaces.msg import LidarDecoded
from message_filters import Subscriber, TimeSynchronizer
from go2_interfaces.msg import LidarDecoded

import zmq
import numpy as np

from lidar_processor.lidar_message_utils import decode_array_from_message


class ROSBridge(Node):
    def __init__(self):
        super().__init__("ros_bridge")

        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PUB)
        self.socket.bind("tcp://localhost:5555")

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        self._setup_subscribers()
        self.sync = TimeSynchronizer(
            [self.decoded_sub, self.filtered_sub],
            queue_size=5
        )
        self.sync.registerCallback(self.synced_cb)

    
    def _setup_subscribers(self) -> None:
        self.decoded_sub = Subscriber(
            self, 
            LidarDecoded,
            "utlidar/decoded_cloud",
            qos_profile=self.qos_profile
        )

        self.filtered_sub = Subscriber(
            self,
            LidarDecoded,
            "utlidar/filtered_cloud",
            qos_profile=self.qos_profile
        )

    
    def send_array(self, array: np.ndarray, topic):
        md = {"dtype": str(array.dtype), "shape": array.shape}

        self.socket.send_string(topic, flags=zmq.SNDMORE)
        self.socket.send_json(md, zmq.SNDMORE)
        self.socket.send(array, copy=False)
        
    
    def synced_cb(self, decoded_msg: LidarDecoded, filtered_msg: LidarDecoded) -> None:
        def combine_xyz_intensity(msg: LidarDecoded) -> np.ndarray:
            arr = decode_array_from_message(msg, "xyz")
            if msg.has_intensity:
                i = decode_array_from_message(msg, "intensity")
                if i.ndim == 1:
                    i = i[:, np.newaxis]
                arr = np.hstack([arr, i])
            return arr

        try:
            d = combine_xyz_intensity(decoded_msg)
            f = combine_xyz_intensity(filtered_msg)

            self.send_array(d, "decoded_topic")
            self.send_array(f, "filtered_topic")

        except Exception as e:
            self.get_logger().error(f"Error processing LiDAR data: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = ROSBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running lidar processor: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()