import sys
import ctypes
import numpy as np
import iceoryx2 as iox2

from iceoryx_interfaces.qos import LidarQoS
from iceoryx_interfaces.lidar_data import LidarHeader_

from .utils.singleton import Singleton
    

class LidarBridge(metaclass=Singleton):
    def __init__(self) -> None:
        iox2.set_log_level_from_env_or(iox2.LogLevel.Info)
        self._node = iox2.NodeBuilder.new() \
                .signal_handling_mode(iox2.SignalHandlingMode.Disabled) \
                .create(iox2.ServiceType.Ipc)

        self._decoded_service = self._node.service_builder(iox2.ServiceName.new(LidarQoS.TOPIC_ROS_LIDAR_DECODED)) \
                                    .publish_subscribe(iox2.Slice[ctypes.c_double]) \
                                    .user_header(LidarHeader_) \
                                    .open_or_create()

        self._filtered_service = self._node.service_builder(iox2.ServiceName.new(LidarQoS.TOPIC_ROS_LIDAR_FILTERED)) \
                                    .publish_subscribe(iox2.Slice[ctypes.c_double]) \
                                    .user_header(LidarHeader_) \
                                    .open_or_create()
        
        self._decoded_pub = self._decoded_service.publisher_builder() \
                                .initial_max_slice_len(2000) \
                                .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo) \
                                .create()

        self._filtered_pub = self._filtered_service.publisher_builder() \
                                .initial_max_slice_len(2000) \
                                .allocation_strategy(iox2.AllocationStrategy.PowerOfTwo) \
                                .create()

    def send_decoded(self, stamp_ns: int, array: np.ndarray) -> None:
            required_memory_size = array.size
            sample = self._decoded_pub.loan_slice_uninit(required_memory_size)
            
            buf = np.asarray(sample.payload())
            ctypes.memmove(buf.ctypes.data, array.ctypes.data, array.nbytes)
            
            rows, cols = array.shape
            sample.user_header().contents.stamp_ns = stamp_ns
            sample.user_header().contents.rows = rows
            sample.user_header().contents.cols = cols

            sample = sample.assume_init()
            sample.send() 

    def send_filtered(self, stamp_ns: int, array: np.ndarray) -> None:
            required_memory_size = array.size
            sample = self._filtered_pub.loan_slice_uninit(required_memory_size)

            buf = np.asarray(sample.payload())
            ctypes.memmove(buf.ctypes.data, array.ctypes.data, array.nbytes)

            rows, cols = array.shape
            sample.user_header().contents.stamp_ns = stamp_ns
            sample.user_header().contents.rows = rows
            sample.user_header().contents.cols = cols

            sample = sample.assume_init()
            sample.send() 