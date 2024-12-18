from __future__ import annotations

import functools
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from .utils import doublewrap


class TimeableMixin:
    """A mixin class to add timing functionality to a class for profiling its methods.

    This mixin class provides the following functionality:
        - Timing of methods using the TimeAs decorator.
        - Timing of arbitrary code blocks using the _time_as context manager.
        - Profiling of the durations of the timed methods.

    Attributes:
        _timings: A dictionary of lists of dictionaries containing the start and end times of timed methods.
            The keys of the dictionary are the names of the timed methods.
            The values are lists of dictionaries containing the start and end times of each timed method call.
            The dictionaries contain the keys "start" and "end" with the corresponding times.
    """

    _START_TIME = "start"
    _END_TIME = "end"

    _CUTOFFS_AND_UNITS = [
        (1000, "μs"),
        (1000, "ms"),
        (60, "sec"),
        (60, "min"),
        (24, "hour"),
        (7, "days"),
        (None, "weeks"),
    ]

    @classmethod
    def _get_pprint_num_unit(cls, x: float, x_unit: str = "sec") -> tuple[float, str]:
        x_unit_factor = 1
        for fac, unit in cls._CUTOFFS_AND_UNITS:
            if unit == x_unit:
                break
            if fac is None:
                raise LookupError(
                    f"Passed unit {x_unit} invalid! "
                    f"Must be one of {', '.join(u for f, u in cls._CUTOFFS_AND_UNITS)}."
                )
            x_unit_factor *= fac

        min_unit = x * x_unit_factor
        upper_bound = 1
        for upper_bound_factor, unit in cls._CUTOFFS_AND_UNITS:
            if (upper_bound_factor is None) or (min_unit < upper_bound * upper_bound_factor):
                return min_unit / upper_bound, unit
            upper_bound *= upper_bound_factor

    @classmethod
    def _pprint_duration(cls, mean_sec: float, n_times: int = 1, std_seconds: float | None = None) -> str:
        mean_time, mean_unit = cls._get_pprint_num_unit(mean_sec)

        if std_seconds:
            std_time = std_seconds * mean_time / mean_sec
            mean_std_str = f"{mean_time:.1f} ± {std_time:.1f} {mean_unit}"
        else:
            mean_std_str = f"{mean_time:.1f} {mean_unit}"

        if n_times > 1:
            return f"{mean_std_str} (x{n_times})"
        else:
            return mean_std_str

    def __init__(self, *args, **kwargs):
        self._timings = kwargs.get("_timings", defaultdict(list))

    def __assert_key_exists(self, key: str) -> None:
        assert hasattr(self, "_timings") and key in self._timings, f"{key} should exist in self._timings!"

    def _times_for(self, key: str) -> list[float]:
        self.__assert_key_exists(key)
        return [
            t[self._END_TIME] - t[self._START_TIME]
            for t in self._timings[key]
            if self._START_TIME in t and self._END_TIME in t
        ]

    def _time_so_far(self, key: str) -> float:
        self.__assert_key_exists(key)
        assert self._END_TIME not in self._timings[key][-1], f"{key} is not currently being timed!"
        return time.time() - self._timings[key][-1][self._START_TIME]

    def _register_start(self, key: str) -> None:
        if not hasattr(self, "_timings"):
            self._timings = defaultdict(list)

        self._timings[key].append({self._START_TIME: time.time()})

    def _register_end(self, key: str) -> None:
        assert hasattr(self, "_timings")
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
    def TimeAs(fn, key: str | None = None):
        if key is None:
            key = fn.__name__

        @functools.wraps(fn)
        def wrapper_timing(self, *args, seed: int | None = None, **kwargs):
            self._register_start(key=key)
            out = fn(self, *args, **kwargs)
            self._register_end(key=key)
            return out

        return wrapper_timing

    @property
    def _duration_stats(self):
        out = {}
        for k in self._timings:
            arr = np.array(self._times_for(k))
            out[k] = (arr.mean(), len(arr), arr.std())
        return out

    def _profile_durations(self, only_keys: set[str] | None = None):
        stats = self._duration_stats

        if only_keys is not None:
            stats = {k: v for k, v in stats.items() if k in only_keys}

        longest_key_length = max(len(k) for k in stats)
        ordered_keys = sorted(stats.keys(), key=lambda k: stats[k][0] * stats[k][1])
        tfk_str = "\n".join(
            (f"{k}:{' '*(longest_key_length - len(k))} " f"{self._pprint_duration(*stats[k])}")
            for k in ordered_keys
        )
        return tfk_str
