from typing import Any, Optional, Tuple

import numpy as np

from src.core.base_module import DogModule
from src.video_control.streamer import WebRTCStreamer
from src.video_control.camera_source import CameraSource


class VideoModule(DogModule):
    """Handles video capture from dog camera or external webcam"""
    def __init__(self, camera_source: CameraSource):
        super().__init__("Video")
        
        self._camera_source = camera_source
        self._streamer = None
        self._streaming = False

        self.initialize()
        

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
            raise RuntimeError("[Video] Stream server already started.")
        
        self._streamer.start_in_thread()
        self._streaming = True

    def send_frame(self, frame: np.ndarray) -> None:
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before sending frames.")

        self._streamer.send(frame)

    def get_stream_server_local_ip(self) -> str:
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before getting its local IP")
        
        return self._streamer.get_local_ip_address()
    
    def get_stream_server_port(self) -> int:
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before getting its port")
        
        return self._streamer.get_port()
        

    def shutdown(self) -> None:
        if not self._initialized:
            return
        
        self._camera_source.shutdown()

        if self._streamer:
            self._streamer._shutdown()
            self._streamer = None

        self._streaming = False
        self._initialized = False
