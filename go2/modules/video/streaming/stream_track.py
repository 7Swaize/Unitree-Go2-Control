import asyncio
import threading
from fractions import Fraction
from typing import Optional
 
import cv2
import numpy as np
from aiortc import VideoStreamTrack
from av import VideoFrame

from .stream_config import StreamConfig

# command to install rtc: pip install aiortc opencv-python

class OpenCVStreamTrack(VideoStreamTrack):
    def __init__(self, stream_config: StreamConfig) -> None:
        super().__init__()
        self._stream_config = stream_config

        yuv_height = stream_config.height * 3 // 2
        self._bufs = [
            np.zeros((yuv_height, stream_config.width), dtype=np.uint8),
            np.zeros((yuv_height, stream_config.width), dtype=np.uint8),
        ]
        self._write_idx: int = 0
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None


    def push_frame(self, bgr_frame: np.ndarray) -> None:
        if bgr_frame is None:
            return
 
        h, w = bgr_frame.shape[:2]
        if w != self._stream_config.width or h != self._stream_config.height:
            bgr_frame = cv2.resize(
                bgr_frame, (self._stream_config.width, self._stream_config.height), interpolation=cv2.INTER_LINEAR
            )
 
        yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)
 
        inactive = 1 - self._write_idx
        np.copyto(self._bufs[inactive], yuv)
        with self._lock:
            self._write_idx = inactive


    async def recv(self) -> VideoFrame:
        loop = asyncio.get_event_loop()
 
        if self._start_time is None:
            self._start_time = loop.time()
 
        elapsed = loop.time() - self._start_time
        frame_idx = int(elapsed * self._stream_config.fps)
 
        with self._lock:
            read_idx = self._write_idx
 
        av_frame = VideoFrame.from_ndarray(self._bufs[read_idx], format="yuv420p")
        av_frame.pts = frame_idx
        av_frame.time_base = Fraction(1, self._stream_config.fps) # weird
        return av_frame