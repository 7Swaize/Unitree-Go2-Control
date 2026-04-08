"""Camera Source Factory"""

from modules.video.camera_source import CameraSource
import threading
import cv2
import numpy as np
from typing import Optional, Tuple, Any
from .frame_buffer import FrameBuffer


class SDKCameraSource(CameraSource):
    '''SDK-based camera source backed by Unitree's VideoClient.'''
    def __init__(self):
        self._video_client = None
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_buffer = FrameBuffer()

    def initialize(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        try:
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
                    
                    if image is not None:
                        self._frame_buffer.put(image)

                except Exception as e:
                    print(f"[SDKCamera] Error in thread: {e}")
                    continue
        except ImportError:
            print("[SDKCamera] Unitree SDK not available")

    def get_frames(self) -> Tuple[int, Optional[np.ndarray]]:
        latest_frame = self._frame_buffer.get()
        if latest_frame is None:
            return -1, None
        
        return 0, latest_frame.copy()
        
    def shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()
        
        if self._thread:
            self._thread.join(timeout=2)


class OpenCVCameraSource(CameraSource):
    """Camera source backed by OpenCV's VideoCapture."""
        
    def __init__(self, camera_index: int = 0):
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
            print(f"[OpenCVCamera] Failed to open camera {self._camera_index}")
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
        latest_frame = self._frame_buffer.get()
        if latest_frame is None:
            return -1, None
        
        return 0, latest_frame.copy()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()
        
        if self._thread:
            self._thread.join(timeout=2)


class RealSenseDepthCamera(CameraSource):
    """
    Depth camera source using Intel RealSense.
    
    This is a simplified stub. Full implementation would handle depth/color alignment.
    """
    
    def __init__(self):
        self._pipeline = None
        self._thread = None
        self._stop_event = threading.Event()
        self._frame_buffer = FrameBuffer()

    def initialize(self) -> None:
        try:
            import pyrealsense2 as rs
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self._pipeline.start(config)
            
            self._thread = threading.Thread(target=self._capture_thread, daemon=True)
            self._thread.start()
        except ImportError:
            print("[RealSense] pyrealsense2 not available")

    def _capture_thread(self):
        try:
            while not self._stop_event.is_set():
                frames = self._pipeline.wait_for_frames()
                color_frame = np.asanyarray(frames.get_color_frame().get_data())
                depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
                self._frame_buffer.put((color_frame, depth_frame))
        except Exception as e:
            print(f"[RealSense] Error in thread: {e}")

    def get_frames(self) -> Tuple[int, Optional[Tuple[np.ndarray, np.ndarray]]]:
        latest_frames = self._frame_buffer.get()
        if latest_frames is None:
            return -1, None
        
        return 0, latest_frames

    def shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()
        
        if self._pipeline:
            self._pipeline.stop()
        
        if self._thread:
            self._thread.join(timeout=2)


class CameraSourceFactory:
    """
    Factory class for creating camera sources.

    Provides a simple, safe way for students to select which camera hardware to use.
    """

    @staticmethod
    def create_sdk_camera() -> CameraSource:
        """Create a camera source backed by the robot's internal camera."""
        return SDKCameraSource()

    @staticmethod
    def create_opencv_camera(camera_index: int = 0) -> CameraSource:
        """Create a camera source using OpenCV's VideoCapture."""
        return OpenCVCameraSource(camera_index)

    @staticmethod
    def create_depth_camera() -> CameraSource:
        """Create an RGB-D camera source using an Intel RealSense device."""
        return RealSenseDepthCamera()
