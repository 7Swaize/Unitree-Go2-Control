from typing import Callable, Dict, TypeVar, Generic, Hashable
from collections import deque

TData = TypeVar("TData")
TKey = TypeVar("TKey", bound=Hashable)


class ExactSynchronizer(Generic[TKey, TData]):
    def __init__(self, callback: Callable[[TKey, TData, TData], None], max_size: int = 5) -> None:
        self._callback = callback
        self._max_size = max_size

        self._buf_l: Dict[TKey, TData] = {}
        self._buf_r: Dict[TKey, TData] = {}
        self._order_l = deque()
        self._order_r = deque()

    def add_left(self, key: TKey, data: TData) -> None:
        if key in self._buf_r:
            r = self._buf_r.pop(key)
            self._callback(key, data, r)
            return

        self._buf_l[key] = data
        self._order_l.append(key)
        self._evict_l()

    def add_right(self, key: TKey, data: TData) -> None:
        if key in self._buf_l:
            l = self._buf_l.pop(key)
            self._callback(key, l, data)
            return
    
        self._buf_r[key] = data
        self._order_r.append(key)
        self._evict_r()

    def _evict_l(self) -> None:
        while len(self._order_l > self._max_size):
            key = self._order_l.popleft()
            if key in self._buf_l:
                del self._buf_l[key] 

    def _evict_r(self) -> None:
        while len(self._order_r > self._max_size):
            key = self._order_r.popleft()
            if key in self._buf_r:
                del self._buf_r[key] 

                