import rclpy
from rclpy.executors import SingleThreadedExecutor

from .lidar_decoder_node import LidarDecoderNode
from .lidar_filter_node import LidarFilterNode


def main(args=None):
    rclpy.init(args=args)

    decoder_node = LidarDecoderNode()
    filter_node = LidarFilterNode()

    executor = SingleThreadedExecutor()
    executor.add_node(decoder_node)
    executor.add_node(filter_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error running lidar processor: {e}")
    finally:
        executor.shutdown()
        decoder_node.destroy_node()
        filter_node.destroy_node()
        rclpy.shutdown()