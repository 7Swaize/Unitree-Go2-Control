from typing import Any, Optional, Tuple

import numpy as np

from unitree_control.core.base_module import DogModule
from unitree_control.video_control.streamer import WebRTCStreamer
from unitree_control.video_control.camera_source import CameraSource, RealSenseDepthCamera, SDKCameraSource, OpenCVCameraSource


class VideoModule(DogModule):
    """Handles video capture from dog camera or external webcam"""
    def __init__(self, camera_source: CameraSource):
        super().__init__("Video")
        
        self._camera_source = camera_source
        self._streamer = None
        self._streaming = False

        self.initialize()

    @classmethod
    def create_sdk_camera(cls) -> CameraSource:
        return SDKCameraSource()
    
    @classmethod
    def create_opencv_camera(cls, camera_index: int = 0) -> CameraSource:
        return OpenCVCameraSource(camera_index)
    
    @classmethod
    def create_depth_camera(cls) -> CameraSource:
        return RealSenseDepthCamera()

    def initialize(self) -> None:
        if self._initialized:
            return
        
        self._camera_source.initialize()
        self._streamer = WebRTCStreamer()
        self._streaming = False
        
        self._initialized = True

    def get_frames(self) -> Tuple[int, Optional[Any]]:
        return self._camera_source.get_frames()


    def start_stream_server(self) -> None:
        if self._streaming:
            print("[Video] Stream Server already started.")
            return
        
        self._streamer.start_in_thread()
        self._streaming = True

    def send_frame(self, frame: np.ndarray) -> None:
        if not self._streaming:
            print("[Video] Stream server not started. Please start the stream server before sending frames.")
            return

        self._streamer.send(frame)

    def get_stream_server_local_ip(self) -> Optional[str]:
        if not self._streaming:
            print("[Video] Stream server not started. Please start the stream server before getting its local IP")
            return None
        
        return self._streamer.get_local_ip_address()

    def shutdown(self) -> None:
        if not self._initialized:
            return
        
        self._camera_source.shutdown()

        if self._streamer:
            self._streamer._shutdown()
            self._streamer = None

        self._streaming = False
        self._initialized = False
