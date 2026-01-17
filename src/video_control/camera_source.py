from abc import ABC, abstractmethod
import threading
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import pyrealsense2 as rs

from src.video_control.frame_buffer import FrameBuffer

class CameraSource(ABC):
    """
    Abstract base class for all camera sources.

    A ``CameraSource`` provides a unified interface for acquiring image frames
    from different camera backends (SDK-based cameras, OpenCV webcams,
    depth cameras, etc.).

    All concrete camera implementations must implement:
        - ``initialize``: start the camera and any background threads
        - ``get_frames``: retrieve the latest available frame(s)
        - ``shutdown``: safely stop the camera and release resources

    This abstraction allows higher-level modules (e.g. VideoModule)
    to consume camera data without caring about the underlying hardware.
    """

    @abstractmethod
    def initialize(self) -> None:
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
    def get_frames(self) -> Tuple[int, Optional[Any]]:
        """
        Retrieve the next frame(s) from the camera source. The amount of frames
        returned are implementation-dependent (e.g., single frame, color+depth
        pair, etc.).

        Returns
        -------
        Tuple[int, Optional[Any]]
            A tuple containing:

                - An integer timestamp or frame index.
                - The frame data, which can be in any format depending on the implementation (e.g., NumPy array, OpenCV frame, etc.). Returns None if no frame is available.

        Notes
        -----
        - This method does **not** block.
        - Implementations typically return the latest frame stored in a buffer.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
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



class SDKCameraSource(CameraSource):
    '''
    SDK-based camera source backed by Unitree's VideoClient.

    This implementation is intended for:
        - Unitree Go2 robot cameras

    Frames are captured in a background thread and stored in a frame buffer.
    '''
    def __init__(self):
        self._video_client = None

        self._thread = None
        self._stop_event = threading.Event()

        self._frame_buffer = FrameBuffer()

    def initialize(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        from unitree_sdk2py.go2.video.video_client import VideoClient
        self._video_client = VideoClient()
        self._video_client.SetTimeout(3.0)
        self._video_client.Init()

        while not self._stop_event.is_set():
            try:
                code, data = self._video_client.GetImageSample()
                if code != 0 or data is None:
                    continue
                
                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                
                self._frame_buffer.put(image)

            except Exception as e:
                print(f"[SDKCamera] Error in thread: {e}")
                continue

    def get_frames(self) -> Tuple[int, Optional[np.ndarray]]:
        '''
        Retrieve the latest frame from the Unitree Go2's internal camera source.

        Returns
        -------
        Tuple[int, Optional[np.ndarray]]
            ``(0, frame)`` if a frame is available
            ``(-1, None)`` if no frame is available.
        '''
        latest_frame = self._frame_buffer.get()
        if latest_frame is None:
            return -1, None
        
        return 0, latest_frame.copy()
        
    def shutdown(self) -> None:
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
        
    def __init__(self, camera_index: int = 0):
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

    def initialize(self) -> None:
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

    def get_frames(self) -> Tuple[int, Optional[np.ndarray]]:
        '''
        Retrieve the latest frame from the OpenCV camera source.

        Returns
        -------
        Tuple[int, Optional[np.ndarray]]
            A tuple containing:

            - ``(0, frame)`` if a frame is available
            - ``(-1, None)`` if no frame is available.
        '''
        latest_frame = self._frame_buffer.get()
        
        if latest_frame is None:
            return -1, None
        
        return 0, latest_frame.copy()
    
    def shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()
        
        if self._thread:
            self._thread.join()
            self._thread = None


# Basic Camera Use: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
# Align: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Optimzation/Configs: https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance

class RealSenseDepthCamera(CameraSource):
    """
    Intel RealSense RGB-D camera source.

    This camera provides **aligned color and depth frames** using
    the RealSense SDK. Frames are captured asynchronously in a
    background thread.

    ``get_frames`` returns a tuple ``(color, depth)`` where:
        - ``color`` is a BGR image (H, W, 3)
        - ``depth`` is a uint16 depth image (H, W)
    """
        
    def __init__(self):
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

    def initialize(self) -> None:
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


    def get_frames(self) -> Tuple[int, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Retrieve the latest aligned color and depth frames.

        Returns
        -------
        Tuple[int, Optional[Tuple[np.ndarray, np.ndarray]]]
            ``(0, (color, depth))`` if frames are available, otherwise
            ``(-1, None)``.
        """
        with self._lock:
            latest_color = self._color_frame_buffer.get()
            latest_depth = self._depth_frame_buffer.get()

            if latest_color is None or latest_depth is None:
                return -1, None
        
            return 0, (latest_color.copy(), latest_depth.copy())


    def shutdown(self) -> None:
        self._stop_event.set()
        self._color_frame_buffer.clear()
        self._depth_frame_buffer.clear()

        if self._thread:
            self._thread.join()
            self._thread = None

