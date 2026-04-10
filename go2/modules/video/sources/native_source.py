import cv2
import numpy as np
import threading
from typing_extensions import override

from unitree_sdk2py.go2.video.video_client import VideoClient

from ..frame_buffer import FrameBuffer
from ..frame_result import FrameResult
from .camera_source import CameraSource


class NativeCameraSource(CameraSource):
    def __init__(self) -> None:
        self._video_client = VideoClient()
        self._video_client.SetTimeout(3.0)
        self._video_client.Init()

        self._thread = None
        self._stop_event = threading.Event()
        self._frame_buffer = FrameBuffer()

    @override
    def _initialize(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        while not self._stop_event.is_set():
            try:
                code, data = self._video_client.GetImageSample()
                if code != 0 or data is None:
                    continue

                image_data = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

                self._frame_buffer.put(image)

            except Exception as e:
                print(f"[NativeCamera] Error in thread: {e}")
                continue

    @override
    def _get_frames(self) -> FrameResult:
        latest_frame = self._frame_buffer.get()
        if latest_frame is None:
            return FrameResult()

        return FrameResult(color=latest_frame.copy())

    @override
    def _shutdown(self) -> None:
        self._stop_event.set()
        self._frame_buffer.clear()

        if self._thread:
            self._thread.join()
            self._thread = None