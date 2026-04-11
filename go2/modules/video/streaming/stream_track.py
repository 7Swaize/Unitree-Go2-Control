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

        self._cfg = stream_config
        yuv_height = stream_config.height * 3 // 2

        self._frame_buf = np.zeros(
            (yuv_height, stream_config.width),
            dtype=np.uint8,
            order="C",
        )

        self._version = 0
        self._latest_version = -1
        self._start_time: Optional[float] = None
        self._loop = asyncio.get_event_loop()

        self._cached_frame: Optional[VideoFrame] = None


    def push_frame(self, bgr_frame: np.ndarray) -> None:
        if bgr_frame is None:
            return

        h, w = bgr_frame.shape[:2]
        if w != self._cfg.width or h != self._cfg.height:
            bgr_frame = cv2.resize(
                bgr_frame,
                (self._cfg.width, self._cfg.height),
                interpolation=cv2.INTER_LINEAR,
            )

        yuv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)
        np.copyto(self._frame_buf, yuv)

        self._version += 1


    async def recv(self) -> VideoFrame:
        if self._start_time is None:
            self._start_time = self._loop.time()

        elapsed = self._loop.time() - self._start_time
        pts = int(elapsed * self._cfg.fps)

        version = self._version

        if version == self._latest_version and self._cached_frame is not None:
            self._cached_frame.pts = pts
            self._cached_frame.time_base = Fraction(1, self._cfg.fps)
            return self._cached_frame

        frame = VideoFrame.from_ndarray(self._frame_buf, format="yuv420p")

        frame.pts = pts
        frame.time_base = Fraction(1, self._cfg.fps)
        self._cached_frame = frame
        self._latest_version = version

        return frame