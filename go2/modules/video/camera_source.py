from abc import ABC, abstractmethod
import threading
from typing_extensions import override

import cv2
import numpy as np
import pyrealsense2 as rs
from unitree_sdk2py.go2.video.video_client import VideoClient

from .frame_buffer import FrameBuffer
from .frame_result import FrameResult

class CameraSource(ABC):
    """
    Abstract base class for all camera sources.

    A ``CameraSource`` provides a unified interface for acquiring image frames
    from different camera backends (SDK-based cameras, OpenCV webcams,
    depth cameras, etc.).

    All concrete camera implementations must implement:
        - ``initialize``: start the camera and any background threads
        - ``get_frames``: retrieve the latest available frame(s) in a :class:`video.FrameResult` object
        - ``shutdown``: safely stop the camera and release resources

    This abstraction allows higher-level modules (e.g. VideoModule)
    to consume camera data without caring about the underlying hardware.
    """

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize the camera source.

        This method should set up any resources required for capturing frames, such as
        opening camera devices, starting streams, or allocating buffers.

        Raises
        ------
        RuntimeError
            If the camera cannot be initialized.
        """
        pass
    
    @abstractmethod
    def _get_frame(self) -> FrameResult:
        """
        Retrieve the next frame from the camera source

        Notes
        -----
        - This method does **not** block.
        - Implementations typically return the latest frame stored in a buffer.
        """
        pass

    @abstractmethod
    def _shutdown(self) -> None:
        """
        Shut down the camera source and release any resources.

        This method should safely close camera devices, stop streams, and clean up
        allocated resources.

        Raises
        ------
        RuntimeError
            If the camera cannot be properly shut down.
        """
        pass



class NativeCameraSource(CameraSource):
    """
    Native camera source backed by Unitree's VideoClient.

    This implementation is intended for:
        - Unitree Go2 robot cameras

    Frames are captured in a background thread and stored in a frame buffer.
    """
    def __init__(self) -> None:
        self._video_client = VideoClient()
        self._video_client.SetTimeout(3.0)
        self._video_client.Init()

        self._thread = None
        self._stop_event = threading.Event()
        self._frame_buffer = FrameBuffer()

    @override
    def _initialize(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        while not self._stop_event.is_set():
            try:
                code, data = self._video_client.GetImageSample()
                if code != 0 or data is None:
                    continue
                
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                self._frame_buffer.put(image)

            except Exception as e:
                print(f"[NativeCamera] Error in thread: {e}")
                continue

    @override
    def _get_frame(self) -> FrameResult:
        """Retrieve the latest frame from the Unitree Go2's internal camera source."""
        latest_frame = self._frame_buffer.get()
        if latest_frame is None:
            return FrameResult()
        
        return FrameResult(color=latest_frame.copy())
        
    @override
    def _shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()
        
        if self._thread:
            self._thread.join()
            self._thread = None


class OpenCVCameraSource(CameraSource):
    """
    Camera source backed by OpenCV's ``VideoCapture``.

    This implementation is intended for:
        - USB webcams
        - Laptop cameras
        - Simple RGB camera setups

    Frames are captured in a background thread and stored in a frame buffer.
    """
        
    def __init__(self, camera_index: int = 0) -> None:
        """
        Parameters
        ----------
        camera_index : int, optional
            OpenCV camera index (default is 0).
        """
        self._capture = None
        self._camera_index = camera_index

        self._thread = None
        self._stop_event = threading.Event()
        self._frame_buffer = FrameBuffer()

    @override
    def _initialize(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        self._capture = cv2.VideoCapture(self._camera_index)
        if not self._capture.isOpened():
            return

        while not self._stop_event.is_set():
            try:
                ret, frame = self._capture.read()
                if not ret:
                    continue
                self._frame_buffer.put(frame)

            except Exception as e:
                print(f"[OpenCVCamera] Error in thread: {e}")
                continue

        self._capture.release()

    @override
    def _get_frame(self) -> FrameResult:
        """Retrieve the latest frame from the OpenCV camera source."""
        latest_frame = self._frame_buffer.get()
        if latest_frame is None:
            return FrameResult()
        
        return FrameResult(color=latest_frame.copy())
    
    @override
    def _shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()
        
        if self._thread:
            self._thread.join()
            self._thread = None


# Basic Camera Use: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
# Align: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Optimzation/Configs: https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance

class RealSenseDepthCameraSource(CameraSource):
    """
    Intel RealSense RGB-D camera source.

    This camera provides **aligned color and depth frames** using the RealSense SDK.
    Frames are captured asynchronously in a background thread.
    """
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
    def _get_frame(self) -> FrameResult:
        """Retrieve the latest aligned color and depth frames."""
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
