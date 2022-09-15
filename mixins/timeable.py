from __future__ import annotations

import functools, time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

from .utils import doublewrap

class TimeableMixin():
    _START_TIME = 'start'
    _END_TIME = 'end'
    def __init__(self, *args, **kwargs):
        self._timings = kwargs.get('_timings', defaultdict(list))

    def __assert_key_exists(self, key: str) -> None:
        assert hasattr(self, '_timings') and key in self._timings, f"{key} should exist in self._timings!"

    def _times_for(self, key: str) -> List[float]:
        self.__assert_key_exists(key)
        return [
            t[self._END_TIME] - t[self._START_TIME] for t in self._timings[key] \
                if self._START_TIME in t and self._END_TIME in t
        ]

    def _time_so_far(self, key: str) -> float:
        self.__assert_key_exists(key)
        assert self._END_TIME not in self._timings[key][-1], f"{key} is not currently being timed!"
        return time.time() - self._timings[key][-1][self._START_TIME]

    def _register_start(self, key: str) -> None:
        if not hasattr(self, '_timings'): self._timings = defaultdict(list)

        self._timings[key].append({self._START_TIME: time.time()})

    def _register_end(self, key: str) -> None:
        assert hasattr(self, '_timings')
        assert key in self._timings and len(self._timings[key]) > 0
        assert self._timings[key][-1].get(self._END_TIME, None) is None
        self._timings[key][-1][self._END_TIME] = time.time()

    @contextmanager
    def _time_as(self, key: str):
        self._register_start(key)
        try:
            yield
        finally:
            self._register_end(key)

    @staticmethod
    @doublewrap
    def TimeAs(fn, key: Optional[str] = None):
        if key is None: key = fn.__name__
        @functools.wraps(fn)
        def wrapper_timing(self, *args, seed: Optional[int] = None, **kwargs):
            self._register_start(key=key)
            out = fn(self, *args, **kwargs)
            self._register_end(key=key)
            return out
        return wrapper_timing
