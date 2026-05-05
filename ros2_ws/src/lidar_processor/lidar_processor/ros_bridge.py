import ctypes
import numpy as np
import iceoryx2 as iox2

from iceoryx_interfaces.lidar_data import LidarHeader_
from iceoryx_interfaces.qos import LidarQoS

from .utils.singleton import Singleton
from .utils.exact_synchronizer import ExactSynchronizer
    

class ROSBridge(metaclass=Singleton):
    def __init__(self) -> None:
        self._sync = ExactSynchronizer[int, np.ndarray](callback=self._publish_synchronized, max_size=5)
        self._init_iox2()

    def _init_iox2(self) -> None:
        iox2.set_log_level_from_env_or(iox2.LogLevel.Info)
        self._node = iox2.NodeBuilder.new() \
                .signal_handling_mode(iox2.SignalHandlingMode.Disabled) \
                .create(iox2.ServiceType.Ipc)

        self._decoded_service = self._node.service_builder(iox2.ServiceName.new(LidarQoS.TOPIC_ROS_LIDAR_DECODED)) \
                                    .publish_subscribe(iox2.Slice[ctypes.c_double]) \
                                    .user_header(LidarHeader_) \
                                    .open_or_create()

        self._filtered_service = self._node.service_builder(iox2.ServiceName.new(LidarQoS.TOPIC_ROS_LIDAR_FILTERED)) \
                                    .public_subscribe(iox2.Slice[ctypes.c_double]) \
                                    .user_header(LidarHeader_) \
                                    .open_or_create()
        
        self._decoded_pub = self._decoded_service.publisher_builder() \
                                .initial_max_slice_len(1000) \
                                .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo) \
                                .create()

        self._filtered_pub = self._decoded_service.publisher_builder() \
                                .initial_max_slice_len(1000) \
                                .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo) \
                                .create()

    def send_decoded(self, stamp_ns: int, array: np.ndarray) -> None:
        self._sync.add_left(stamp_ns, array)

    def send_filtered(self, stamp_ns: int, array: np.ndarray) -> None:
        self._sync.add_right(stamp_ns, array)

    def _publish_synchronized(self, stamp_ns: int, decoded: np.ndarray, filtered: np.ndarray) -> None:
        def pd() -> None:
            required_memory_size = decoded.size
            sample = self._decoded_pub.loan_slice_uninit(required_memory_size)
            
            buf = np.asarray(sample.payload())
            ctypes.memmove(buf.ctypes.data, decoded.ctypes.data, decoded.nbytes)
            
            rows, cols = decoded.shape
            sample.user_header().contents.stamp_ns = stamp_ns
            sample.user_header().contents.rows = rows
            sample.user_header().contents.cols = cols

            sample = sample.assume_init()
            sample.send() 

        def ps() -> None:
            required_memory_size = filtered.size
            sample = self._filtered_pub.loan_slice_uninit(required_memory_size)

            buf = np.asarray(sample.payload())
            ctypes.memmove(buf.ctypes.data, filtered.ctypes.data, filtered.nbytes)

            rows, cols = decoded.shape
            sample.user_header().contents.stamp_ns = stamp_ns
            sample.user_header().contents.rows = rows
            sample.user_header().contents.cols = cols

            sample = sample.assume_init()
            sample.send() 

        pd()
        ps()