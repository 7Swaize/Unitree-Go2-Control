import threading
from typing_extensions import override
import numpy as np
import pyrealsense2 as rs

from ..frame_buffer import FrameBuffer
from ..frame_result import FrameResult
from .camera_source import CameraSource

# Basic Camera Use: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
# Align: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Optimzation/Configs: https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance

class RealSenseDepthCameraSource(CameraSource):
    def __init__(self) -> None:
        self._width: int = 848
        self._height: int = 480
        self._fps: int = 30

        self._pipeline = None
        self._align = None

        self._thread = None
        self._lock = threading.Lock() # need lock because there are 2 frame buffers
        self._stop_event = threading.Event()

        self._color_frame_buffer = FrameBuffer()
        self._depth_frame_buffer = FrameBuffer()

    @override
    def _initialize(self) -> None:
        self._thread = threading.Thread(target=self._rs_thread, daemon=True)
        self._thread.start()

    def _rs_thread(self):
        self._pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.depth, self._width, self._height, rs.format.z16, self._fps)
        config.enable_stream(rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps)

        self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)

        while not self._stop_event.is_set():
            try:
                frames = self._pipeline.wait_for_frames()
                aligned = self._align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                with self._lock:
                    self._color_frame_buffer.put(np.asanyarray(color_frame.get_data()))
                    self._depth_frame_buffer.put(np.asanyarray(depth_frame.get_data()))

            except Exception as e:
                print(f"[RealSense] Error in thread: {e}")
                continue

        self._pipeline.stop()


    @override
    def _get_frames(self) -> FrameResult:
        with self._lock:
            latest_color = self._color_frame_buffer.get()
            latest_depth = self._depth_frame_buffer.get()
            if latest_color is None or latest_depth is None:
                return FrameResult()

            return FrameResult(color=latest_color, depth=latest_depth)


    @override
    def _shutdown(self) -> None:
        self._stop_event.set()
        self._color_frame_buffer.clear()
        self._depth_frame_buffer.clear()

        if self._thread:
            self._thread.join()
            self._thread = None