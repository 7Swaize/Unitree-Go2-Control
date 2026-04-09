import numpy as np
import threading

from typing import Optional


class FrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
 
    def put(self, item: np.ndarray) -> None:
        with self._lock:
            self._frame = item
 
    def get(self) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame
 
    def peek(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame
 
    def clear(self) -> None:
        with self._lock:
            self._frame = None