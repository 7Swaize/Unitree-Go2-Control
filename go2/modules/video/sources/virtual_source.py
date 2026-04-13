import cv2
import threading
import numpy as np
import iceoryx2 as iox2
from typing import Optional
from typing_extensions import override
from iceoryx_interfaces.camera_data import DepthFrameData_, RGBFrameData_
from iceoryx_interfaces.qos import CameraQoS

from .camera_source import CameraSource
from ..frame_result import FrameResult

class VirtualCameraSource(CameraSource):
    def __init__(self) -> None:
        self._thread = None
        self._stop_event = threading.Event()
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._initialize_iox_services()

    @override
    def _start(self) -> None:
        self._thread = threading.Thread(target=self._iox_thread, daemon=True)
        self._thread.start()

    def _initialize_iox_services(self) -> None:
        iox2.set_log_level_from_env_or(iox2.LogLevel.Error)
        
        # Kill signal handling: https://github.com/eclipse-iceoryx/iceoryx2/issues/528
        self._node = iox2.NodeBuilder.new() \
                        .signal_handling_mode(iox2.SignalHandlingMode.Disabled) \
                        .create(iox2.ServiceType.Ipc)
        
        self._depth_service = self._node.service_builder(iox2.ServiceName.new(CameraQoS.TOPIC_SIM_CAMERA_DEPTH)) \
                                .publish_subscribe(DepthFrameData_) \
                                .max_publishers(CameraQoS.MAX_PUBLISHERS) \
                                .max_subscribers(CameraQoS.MAX_SUBSCRIBERS) \
                                .subscriber_max_buffer_size(CameraQoS.SUBSCRIBER_MAX_BUFFER_SIZE) \
                                .subscriber_max_borrowed_samples(CameraQoS.SUBSCRIBER_MAX_BORROWED_SAMPLES) \
                                .history_size(CameraQoS.HISTORY_SIZE) \
                                .open_or_create()
        
        self._rgb_service = self._node.service_builder(iox2.ServiceName.new(CameraQoS.TOPIC_SIM_CAMERA_RGB)) \
                                .publish_subscribe(RGBFrameData_) \
                                .max_publishers(CameraQoS.MAX_PUBLISHERS) \
                                .max_subscribers(CameraQoS.MAX_SUBSCRIBERS) \
                                .subscriber_max_buffer_size(CameraQoS.SUBSCRIBER_MAX_BUFFER_SIZE) \
                                .subscriber_max_borrowed_samples(CameraQoS.SUBSCRIBER_MAX_BORROWED_SAMPLES) \
                                .history_size(CameraQoS.HISTORY_SIZE) \
                                .open_or_create()

        self._depth_sub = self._depth_service.subscriber_builder().create()
        self._rgb_sub = self._rgb_service.subscriber_builder().create()
        self._cycle_time = iox2.Duration.from_millis(1)            


    def _iox_thread(self):
        while not self._stop_event.is_set():
            self._node.wait(self._cycle_time)

            while True:
                sample = self._depth_sub.receive()
                if sample is None:
                    break

                msg = sample.payload().contents
                self._add_depth_to_buffer(
                    np.array(msg.data, copy=True, dtype=np.float32).reshape((msg.height, msg.width)),
                    msg.depth_min,
                    msg.depth_max
                )

            while True:
                sample = self._rgb_sub.receive()
                if sample is None:
                    break

                msg = sample.payload().contents
                self._add_rgb_to_buffer(
                    np.array(msg.data, copy=True, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                )

    def _add_rgb_to_buffer(self, rgb_np: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(rgb_np[::-1], cv2.COLOR_RGB2BGR)
        self._latest_rgb = rgb_frame

    def _add_depth_to_buffer(self, depth_np: np.ndarray, depth_min: float, depth_max: float) -> None:
        depth_frame = self._depth_to_colormap(depth_np[::-1], depth_min, depth_max)
        self._latest_depth = depth_frame

    def _depth_to_colormap(self, depth_np: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
        rng = depth_max - depth_min or 1.0
        alpha = 255.0 / rng
        beta = -depth_min * alpha

        u8 = cv2.convertScaleAbs(depth_np, alpha=alpha, beta=beta)
        colormap = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO) # TODO: Maybe its COLORMAP_JET -> see what d345i uses.

        # Black out pixels that carried no valid depth return.
        colormap[(depth_np <= 0.0) | ~np.isfinite(depth_np)] = 0

        return colormap

    @override
    def _get_frames(self) -> FrameResult:
        if self._latest_rgb is None or self._latest_depth is None:
            return FrameResult.pending()

        return FrameResult.color_and_depth(color=self._latest_rgb.copy(), depth=self._latest_depth.copy())

    @override
    def _shutdown(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None