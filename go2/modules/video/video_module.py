from typing import Union
from typing_extensions import override, overload
import numpy as np

from ...core.module import DogModule
from .streaming.streamer import WebRTCStreamer
from .camera_source import CameraSource
from .camera_group import CameraGroup
from .frame_result import FrameResult, MultiFrameResult


class VideoModule(DogModule):
    """
    High-level video interface for users.

    ``VideoModule`` provides a simple API for:
        - Accessing camera frames
        - Streaming video to a browser via WebRTC

    Users should interact **only** with this class and should not
    directly use camera sources, streamers, or threading utilities.

    Parameters
    ----------
    camera_source : CameraSource or CameraGroup
        - A :class:`~modules.video.camera_source.CameraSource` instance provided by :class:`~modules.video.camera_source_factory.CameraSourceFactory`
        - A :class:`~modules.video.camera_group.CameraGroup` instance provided by :class:`~modules.video.camera_source_factory.CameraSourceFactory`

    Notes
    -----
    - Camera initialization is handled automatically.
    - Frames are retrieved using :meth:`get_frame`.
    - Streaming must be explicitly started before sending frames.
    """

    def __init__(self, camera_source: Union[CameraSource, CameraGroup]) -> None:
        super().__init__("Video")
        self._camera_source = camera_source
        self._streamer = None
        self._streaming = False
        

    @override
    def _initialize(self) -> None:
        """
        Initialize the video module. This is called internally,
        and should not be called directly by users.

        This method initializes the underlying camera and prepares
        the streaming system. It is called automatically during
        construction and does not need to be called by users.
        """
        if self._initialized:
            return
        
        self._camera_source._initialize()
        self._streamer = WebRTCStreamer()
        self._streaming = False
        self._initialized = True


    def get_frames(self) -> Union[FrameResult, MultiFrameResult]:
        """
        Retrieve the latest frame(s) from all cameras. ``FrameResult`` objects will be empty 
        if no frame was currently available.

        Returns
        -------
        FrameResult
            If constructed with a single ``CameraSource``.
        MultiFrameResult
            If constructed with a ``CameraGroup``.
        """
        return self._camera_source._get_frames()


    def start_stream_server(self) -> None:
        """
        Start the video streaming server.

        This launches a local WebRTC server that allows video frames to be viewed in a browser.
        Only clients connected to the same subnet as the robot can access the stream.

        Raises
        ------
        RuntimeError
            If the stream server has already been started.
        """
        if self._streaming:
            raise RuntimeError("[Video] Stream server already started.")

        self._streamer._start_in_thread()
        self._streaming = True

    def send_frame(self, frame: np.ndarray) -> None:
        """
        Send a video frame to connected streaming clients.

        Parameters
        ----------
        frame : numpy.ndarray
            An image frame to stream (typically obtained from
            :meth:`get_frame`).

        Raises
        ------
        RuntimeError
            If the stream server has not been started.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before sending frames.")

        self._streamer._send(frame)

    def get_stream_server_local_ip(self) -> str:
        """
        Get the local IP address of the stream server.

        Raises
        ------
        RuntimeError
            If the stream server is not running.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before getting its local IP")
        
        return self._streamer._get_local_ip_address()
    
    def get_stream_server_port(self) -> int:
        """
        Get the port number of the stream server.

        Raises
        ------
        RuntimeError
            If the stream server is not running.
        """
        if not self._streaming:
            raise RuntimeError("[Video] Stream server not started. Please start the stream server before getting its port")
        
        return self._streamer._get_port()
        

    @override
    def _shutdown(self) -> None:
        """
        Shut down the video module. This is handled automatically and shouldn't be called by users.

        This stops camera capture, shuts down the streaming server,
        and releases all resources. 
        """
        if not self._initialized:
            return
        
        self._camera_source._shutdown()

        if self._streamer:
            self._streamer._shutdown()
            self._streamer = None

        self._streaming = False
        self._initialized = False