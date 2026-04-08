"""Frame Buffer for Video Streaming"""

import threading
from collections import deque
from typing import Optional, Any
import numpy as np


class FrameBuffer:
    """Thread-safe frame buffer for camera frames."""
    
    def __init__(self, max_size: int = 2):
        """
        Initialize the frame buffer.

        Parameters
        ----------
        max_size : int, optional
            Maximum number of frames to buffer (default is 2).
        """
        self._buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def put(self, frame: Any) -> None:
        """Add a frame to the buffer."""
        with self._lock:
            self._buffer.append(frame)

    def get(self) -> Optional[Any]:
        """Get the most recent frame from the buffer."""
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer[-1]
            return None

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()

    def size(self) -> int:
        """Get the current number of frames in the buffer."""
        with self._lock:
            return len(self._buffer)
