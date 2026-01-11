import numpy as np
import queue

from typing import Optional


class FrameBuffer:
    def __init__(self, max_size: int = 10):
        self._queue = queue.Queue(maxsize=max_size)


    def put(self, item: np.ndarray) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            _ = self._queue.get_nowait()
            self._queue.put_nowait(item)

    def get(self) -> Optional[np.ndarray]:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def get_batch(self, batch_size: int) -> list[np.ndarray]:
        items = []

        for _ in range(batch_size):
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break

        return items

    def clear(self) -> None:
        while not self._queue.empty():
            _ = self._queue.get_nowait()