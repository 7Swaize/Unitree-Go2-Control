"""
Video Module for Student Use
============================

This module provides a high-level interface for accessing camera frames
and streaming video from the robot or an attached webcam. Students should
interact only with :class:`VideoModule` and should not use lower-level
camera or streaming classes directly.

Students should not access or construct this class directly. Rather, they should access it through the :class:`~src.core.unitree_control_core.UnitreeGo2Controller` instance.

Example
-------
>>> from src.core.unitree_control_core import UnitreeGo2Controller
>>> from src.core.module_registry import ModuleType
>>> from src.video_control.camera_source_factory import CameraSourceFactory
>>>
>>> unitree_controller = UnitreeGo2Controller(sdk_enabled=True)
>>> unitree_controller.add_module(ModuleType.VIDEO, camera_source=CameraSourceFactory.create_depth_camera()) # Add video module with depth camera
>>> unitree_controller.video.start_stream_server()  # Start streaming server
>>> while True:
...    status, frames = unitree_controller.video.get_frames()
...    if status != 0:
...        continue
...
...    color_frame, depth_frame = frames
...    unitree_controller.video.send_frame(color_frame)  # Send color frame to stream
"""


from typing import Any, Optional, Tuple

import numpy as np

from src.core.base_module import DogModule
from src.video_control.streamer import WebRTCStreamer
from src.video_control.camera_source import CameraSource


class VideoModule(DogModule):
    """
    High-level video interface for students.

    ``VideoModule`` provides a simple API for:
        - Accessing camera frames
        - Streaming video to a browser via WebRTC

    Students should interact **only** with this class and should not
    directly use camera sources, streamers, or threading utilities.

    Parameters
    ----------
    camera_source : CameraSource
        A camera source provided by the system (e.g. dog camera,
        webcam, or depth camera). Requires a :class:`~src.video_control.camera_source.CameraSource` instance.

    Notes
    -----
    - Camera initialization is handled automatically.
    - Frames are retrieved using :meth:`get_frames`.
    - Streaming must be explicitly started before sending frames.
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

        This method initializes the underlying camera and prepares
        the streaming system. It is called automatically during
        construction and does not need to be called by students.
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

        Examples
        --------
        Single-frame usage (most cameras):

        >>> status, frame = unitree_controller.video.get_frames()
        >>> if status == 0 and isinstance(frame, np.ndarray):
        ...     print(frame.shape)

        Depth-camera usage:

        >>> status, frames = unitree_controller.video.get_frames()
        >>> if status == 0 and isinstance(frames, tuple):
        ...     color, depth = frames
        ...     print(color.shape, depth.shape)
        """
        return self._camera_source.get_frames()


    def start_stream_server(self) -> None:
        """
        Start the video streaming server.

        This launches a local WebRTC server that allows video frames
        to be viewed in a browser. Only clients connected to the same
        subnet as the robot can access the stream.

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
            An image frame to stream (typically obtained from
            :meth:`get_frames`).

        Raises
        ------
        RuntimeError
            If the stream server has not been started.

        Examples
        --------
        >>> status, frame = unitree_controller.video.get_frames()
        >>> if status == 0:
        ...     unitree_controller.video.send_frame(frame)
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

        Raises
        ------
        RuntimeError
            If the stream server is not running.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before getting its local IP")
        
        return self._streamer.get_local_ip_address()
    
    def get_stream_server_port(self) -> int:
        """
        Get the port number of the stream server.

        Returns
        -------
        int
            Port number used by the stream server.

        Raises
        ------
        RuntimeError
            If the stream server is not running.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before getting its port")
        
        return self._streamer.get_port()
        

    def shutdown(self) -> None:
        """
        Shut down the video module. This is handled automatically and shouldn't be called by students.

        This stops camera capture, shuts down the streaming server,
        and releases all resources. This should be called when the
        program is exiting.
        """
        if not self._initialized:
            return
        
        self._camera_source.shutdown()

        if self._streamer:
            self._streamer.shutdown()
            self._streamer = None

        self._streaming = False
        self._initialized = False
