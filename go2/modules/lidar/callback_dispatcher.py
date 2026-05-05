import threading
import queue
import numpy as np

from typing import Any, Callable
from typing_extensions import override

from .utils.exact_synchronizer import ExactSynchronizer


class CallbackDispatcher:
    def __init__(self) -> None:
        self._decoded_callbacks: list[Callable[[int, np.ndarray], None]] = []
        self._filtered_callbacks: list[Callable[[int, np.ndarray], None]] = []
        self._sync_callbacks: list[Callable[[int, np.ndarray, np.ndarray], None]]

        self._sync = ExactSynchronizer[int, np.ndarray](self._emit_synced, max_size=5)


    def _register_decoded(self, cb: Callable[[int, np.ndarray], None]) -> None:
        self._decoded_callbacks.append(cb)

    def _register_filtered(self, cb: Callable[[int, np.ndarray], None]) -> None:
        self._filtered_callbacks.append(cb)

    def _register_synced(self, cb: Callable[[int, np.ndarray, np.ndarray], None]) -> None:
        self._sync_callbacks.append(cb)

    def _emit_decoded(self, stamp_ns: int, array: np.ndarray) -> None:
        self._sync.add_left(stamp_ns, array)
        
        for cb in self._decoded_callbacks:
            cb(stamp_ns, array)

    def _emit_filtered(self, stamp_ns: int, array: np.ndarray) -> None:
        self._sync.add_right(stamp_ns, array)
        
        for cb in self._filtered_callbacks:
            cb(stamp_ns, array)

    def _emit_synced(self, stamp_ns: int, decoded: np.ndarray, filtered: np.ndarray) -> None:
        for cb in self._filtered_callbacks:
            cb(stamp_ns, decoded, filtered)