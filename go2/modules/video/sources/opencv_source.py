import cv2
import threading
from typing_extensions import override

from ..frame_buffer import FrameBuffer
from ..frame_result import FrameResult
from .camera_source import CameraSource


class OpenCVCameraSource(CameraSource):
    def __init__(self, camera_index: int = 0) -> None:
        self._capture = None
        self._camera_index = camera_index

        self._thread = None
        self._stop_event = threading.Event()
        self._frame_buffer = FrameBuffer()

    @override
    def _initialize(self) -> None:
        self._thread = threading.Thread(target=self._capture_thread, daemon=True)
        self._thread.start()

    def _capture_thread(self):
        self._capture = cv2.VideoCapture(self._camera_index)
        if not self._capture.isOpened():
            return

        while not self._stop_event.is_set():
            try:
                ret, frame = self._capture.read()
                if not ret:
                    continue
                self._frame_buffer.put(frame)

            except Exception as e:
                print(f"[OpenCVCamera] Error in thread: {e}")
                continue

        self._capture.release()

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