from abc import ABC, abstractmethod
import threading
from typing import Optional, Tuple, Any

import cv2
import numpy as np
import pyrealsense2 as rs

from unitree_control.video_control.frame_buffer import FrameBuffer


class CameraSource(ABC):
    @abstractmethod
    def initialize(self) -> None:
        pass
    
    @abstractmethod
    def get_frames(self) -> Tuple[int, Optional[Any]]:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


class CameraSourceFactory:  
    @staticmethod
    def create_sdk_camera() -> CameraSource:
        return SDKCameraSource()
    
    @staticmethod
    def create_opencv_camera(camera_index: int = 0) -> CameraSource:
        return OpenCVCameraSource(camera_index)
    
    @staticmethod
    def create_depth_camera() -> CameraSource:
        return RealSenseDepthCamera()
    

class SDKCameraSource(CameraSource):
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
            self._thread.join()
            self._thread = None


# Basic Camera Use: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py
# Align: https://github.com/realsenseai/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
# Optimzation/Configs: https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance

class RealSenseDepthCamera(CameraSource):
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

