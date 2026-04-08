"""Video Module for Student Use"""

from typing import Any, Optional, Tuple
import numpy as np

from core.module import DogModule
from modules.video.streamer import WebRTCStreamer
from modules.video.camera_source import CameraSource


class VideoModule(DogModule):
    """
    High-level video interface for students.

    ``VideoModule`` provides a simple API for:
        - Accessing camera frames
        - Streaming video to a browser via WebRTC

    Parameters
    ----------
    camera_source : CameraSource
        A camera source provided by the system (e.g. dog camera,
        webcam, or depth camera).
    """
    def __init__(self, camera_source: CameraSource):
        super().__init__("Video")
        
        self._camera_source = camera_source
        self._streamer = None
        self._streaming = False
        

    def initialize(self) -> None:
        """
        Initialize the video module. This is called internally,
        and should not be called directly by students.
        """
        if self._initialized:
            return
        
        self._camera_source.initialize()
        self._streamer = WebRTCStreamer()
        self._streaming = False
        
        self._initialized = True

    def get_frames(self) -> Tuple[int, Optional[Any]]:
        """
        Retrieve the latest camera frame(s).

        Depending on the camera type, this method may return:
            - A single color image (RGB/BGR)
            - A tuple ``(color_frame, depth_frame)`` for depth cameras

        Returns
        -------
        Tuple[int, Optional[Any]]
            A tuple ``(status, frames)`` where:
                - ``status`` is ``0`` if frame data is available
                - ``status`` is ``-1`` if no frame data is available
                - ``frames`` is one of:
                    * ``numpy.ndarray`` — a single color image
                    * ``Tuple[numpy.ndarray, numpy.ndarray]`` — ``(color, depth)``
                    * ``None`` — if no frame is available
        """
        return self._camera_source.get_frames()


    def start_stream_server(self) -> None:
        """
        Start the video streaming server.

        This launches a local WebRTC server that allows video frames
        to be viewed in a browser.

        Raises
        ------
        RuntimeError
            If the stream server has already been started.
        """
        if self._streaming:
            raise RuntimeError("[Video] Stream server already started.")
        
        self._streamer.start_in_thread()
        self._streaming = True

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Send a video frame to connected streaming clients.

        Parameters
        ----------
        frame : numpy.ndarray
            An image frame to stream.

        Raises
        ------
        RuntimeError
            If the stream server has not been started.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before sending frames.")

        self._streamer.send(frame)

    def get_stream_server_local_ip(self) -> str:
        """
        Get the local IP address of the stream server.

        Returns
        -------
        str
            Local IP address where the stream is hosted.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started.")
        
        return self._streamer.get_local_ip_address()
    
    def get_stream_server_port(self) -> int:
        """
        Get the port number of the stream server.

        Returns
        -------
        int
            Port number used by the stream server.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started.")
        
        return self._streamer.get_port()

    def shutdown(self) -> None:
        """Shut down the video module."""
        if self._streamer:
            self._streamer.shutdown()
        
        if self._camera_source:
            self._camera_source.shutdown()
        
        self._initialized = False


__all__ = ["VideoModule"]
