from aiortc import VideoStreamTrack
from av import VideoFrame

import numpy as np
import queue


# command to install rtc: pip install aiortc opencv-python

class OpenCVStreamTrack(VideoStreamTrack):
    """
    Video stream track that supplies frames from a local queue to WebRTC.

    This class wraps a `queue.Queue` of frames and presents them as
    an `aiortc.VideoStreamTrack`. It is designed for internal use
    by `WebRTCStreamer`.

    Parameters
    ----------
    frame_queue : queue.Queue
        A thread-safe queue containing frames to stream.
    """
    def __init__(self, frame_queue):
        super().__init__()
        self._frame_queue = frame_queue
        self._last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._frame_count = 0

    async def recv(self):
        """
        Receive the next video frame for WebRTC.

        Retrieves the latest frame from the internal queue. If no
        frame is available, the last frame is repeated.
        """
        pts, time_base = await self.next_timestamp()

        frame = None
        while True:
            try:
                frame = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        if frame is not None:
            self._last_frame = frame
            self._frame_count += 1

        frame = self._last_frame.copy()

        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame
