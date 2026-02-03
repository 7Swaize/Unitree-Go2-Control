import threading
import queue
import numpy as np
from typing import Any, Callable


class CallbackDispatcher(threading.Thread):
    def __init__(self):
        super.__init__(daemon=True)
        self._decoded_callbacks: list[Callable[[int, np.ndarray], Any]] = []
        self._filtered_callbacks: list[Callable[[int, np.ndarray], Any]] = []

        self._decoded_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=5)
        self._filtered_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue(maxsize=5)

        self._running = threading.Event()
        self._running.set()


    def register_decoded(self, cb: Callable[[int, np.ndarray], Any]) -> None:
        self._decoded_callbacks.append(cb)


    def register_filtered(self, cb: Callable[[int, np.ndarray], Any]) -> None:
        self._filtered_callbacks.append(cb)

    
    def emit(self, topic: str, stamp: int, array: np.ndarray) -> None:
        try:
            if topic == "decoded_topic":
                self._decoded_queue.put_nowait((stamp, array))
            elif topic == "filtered_topic":
                self._filtered_queue.put_nowait((stamp, array))
        except queue.Full:
            pass # I currently just drop because of backlog


    def run(self) -> None:
        while self._running.is_set():
            self._drain_queue(self._decoded_queue, self._decoded_callbacks)
            self._drain_queue(self._filtered_queue, self._filtered_callbacks)


    def _drain_queue(self, q: queue.Queue, callbacks: list[Callable[[int, np.ndarray], Any]]) -> None:
        try:
            stamp, arr = q.get(timeout=0.05)
            for cb in callbacks:
                try:
                    cb(stamp, arr)
                except Exception as e:
                    print(f"[CallbackDispatcher] callback error: {e}")

        except queue.Empty:
            pass


    def shutdown(self) -> None:
        self._running.clear()

