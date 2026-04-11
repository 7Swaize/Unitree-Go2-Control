import cv2
import threading
import numpy as np
from typing import Optional
from typing_extensions import override

from unitree_sdk2py.go2.video.video_client import VideoClient

from ..frame_result import FrameResult
from .camera_source import CameraSource


class NativeCameraSource(CameraSource):
    def __init__(self) -> None:
        self._video_client = VideoClient()
        self._video_client.SetTimeout(3.0)
        self._video_client.Init()

        self._thread = None
        self._stop_event = threading.Event()
        self._latest_rgb: Optional[np.ndarray] = None

    @override
    def _start(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        while not self._stop_event.is_set():
            code, data = self._video_client.GetImageSample()
            if code != 0 or data is None:
                continue

            image_data = np.frombuffer(bytes(data), dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

            self._latest_rgb = image


    @override
    def _get_frames(self) -> FrameResult:
        if self._latest_rgb is None:
            return FrameResult.pending()

        return FrameResult.color_only(self._latest_rgb.copy())
    
    @override
    def _shutdown(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join()
            self._thread = None