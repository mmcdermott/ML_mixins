from __future__ import annotations

import functools
import json
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from memray import Tracker

from .utils import doublewrap, pprint_stats_map


class MemTrackableMixin:
    """A mixin class to add memory tracking functionality to a class for profiling its methods.

    This mixin class provides the following functionality:
        - Tracking the memory use of methods using the TrackMemoryAs decorator.
        - Timing of arbitrary code blocks using the _track_memory_as context manager.
        - Profiling of the memory used across the life cycle of the class.

    This class uses `memray` to track the memory usage of the methods.

    Attributes:
        _memory_usage: A dictionary of lists of memory usages of tracked methods.
            The keys of the dictionary are the names of the tracked code blocks / methods.
            The values are lists of memory used during the lifecycle of each tracked code blocks / methods.
    """

    _CUTOFFS_AND_UNITS = [
        (8, "b"),
        (1000, "B"),
        (1000, "kB"),
        (1000, "MB"),
        (1000, "GB"),
        (1000, "TB"),
        (1000, "PB"),
    ]

    @staticmethod
    def get_memray_stats(memray_tracker_fp: Path, memray_stats_fp: Path) -> dict:
        """Extracts the memory stats from the tracker file and saves it to a json file and returns the stats.

        Args:
            memray_tracker_fp: The path to the memray tracker file.
            memray_stats_fp: The path to save the memory statistics.

        Returns:
            The stats extracted from the memray tracker file.

        Raises:
            FileNotFoundError: If the memray tracker file is not found.
            ValueError: If the ``memray stats`` subprocess fails (e.g. malformed tracker file,
                missing ``memray`` binary) — the underlying stderr is surfaced in the message — or
                if the resulting stats JSON cannot be parsed.

        Examples:
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     memray_tracker_fp = Path(tmpdir) / ".memray"
            ...     memray_stats_fp = Path(tmpdir) / "memray_stats.json"
            ...     with Tracker(memray_tracker_fp, follow_fork=True):
            ...         A = np.ones((1000,), dtype=np.float64)
            ...     stats = MemTrackableMixin.get_memray_stats(memray_tracker_fp, memray_stats_fp)
            ...     print(stats['metadata']['peak_memory'])
            8000

        If you run it on a tracker file that is malformed, it won't work — the underlying
        ``memray stats`` error is surfaced in the ``ValueError`` rather than swallowed:
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     memray_tracker_fp = Path(tmpdir) / ".memray"
            ...     memray_stats_fp = Path(tmpdir) / "memray_stats.json"
            ...     memray_tracker_fp.touch()
            ...     MemTrackableMixin.get_memray_stats(memray_tracker_fp, memray_stats_fp)
            Traceback (most recent call last):
                ...
            ValueError: `memray stats` failed for ...

        If you run it on a non-existent file, it won't work.
            >>> MemTrackableMixin.get_memray_stats(Path("non_existent.mem"), Path("non_existent.stats"))
            Traceback (most recent call last):
                ...
            FileNotFoundError: Memray tracker file not found at non_existent.mem
        """
        if not memray_tracker_fp.is_file():
            raise FileNotFoundError(f"Memray tracker file not found at {memray_tracker_fp}")

        try:
            subprocess.run(
                [
                    "memray",
                    "stats",
                    str(memray_tracker_fp),
                    "--json",
                    "-o",
                    str(memray_stats_fp),
                    "-f",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip() or "No stderr output."
            raise ValueError(f"`memray stats` failed for {memray_tracker_fp}:\n{stderr}") from e
        try:
            return json.loads(memray_stats_fp.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse memray stats JSON at {memray_stats_fp}") from e

    def __init__(self, *args, **kwargs):
        self._mem_stats = kwargs.get("_mem_stats", defaultdict(list))

    def __assert_key_exists(self, key: str) -> None:
        if not hasattr(self, "_mem_stats"):
            raise AttributeError("self._mem_stats should exist!")
        if key not in self._mem_stats:
            raise AttributeError(f"{key} should exist in self._mem_stats!")

    def _peak_mem_for(self, key: str) -> list[float]:
        """Return per-call peak memory (in bytes) recorded under ``key``."""
        self.__assert_key_exists(key)

        return [v["metadata"]["peak_memory"] for v in self._mem_stats[key]]

    @contextmanager
    def _track_memory_as(self, key: str):
        """Context manager that tracks peak memory of the enclosed block under ``key``.

        Runs a ``memray.Tracker`` against a temporary file, parses its stats, and appends the result to
        ``self._mem_stats[key]``. The tracker file and its stats export are cleaned up on exit.
        """
        if not hasattr(self, "_mem_stats"):
            self._mem_stats = defaultdict(list)

        memory_stats = {}
        with TemporaryDirectory() as tmpdir:
            memray_fp = Path(tmpdir) / ".memray"
            memray_stats_fp = Path(tmpdir) / "memray_stats.json"

            try:
                with Tracker(memray_fp, follow_fork=True):
                    yield
            finally:
                memory_stats.update(MemTrackableMixin.get_memray_stats(memray_fp, memray_stats_fp))
                self._mem_stats[key].append(memory_stats)

    @staticmethod
    @doublewrap
    def TrackMemoryAs(fn, key: str | None = None):
        """Decorator that tracks peak memory per call to ``fn`` under ``key`` (default: ``fn.__name__``).

        Use as ``@TrackMemoryAs`` or ``@TrackMemoryAs(key="custom")``. See ``_track_memory_as`` for the
        context-manager equivalent.
        """
        if key is None:
            key = fn.__name__

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            with self._track_memory_as(key):
                out = fn(self, *args, **kwargs)
            return out

        return wrapper

    @property
    def _memory_stats(self):
        """Per key: ``(mean_peak_bytes, count, std_or_None)``; ``std`` omitted for single-call keys."""
        out = {}
        for k in self._mem_stats:
            arr = np.array(self._peak_mem_for(k))
            out[k] = (arr.mean(), len(arr), None if len(arr) <= 1 else arr.std())
        return out

    def _profile_memory_usages(self, only_keys: set[str] | None = None):
        """Render ``_memory_stats`` as a multi-line aligned string, with auto-selected units per key.

        Rows are sorted ascending by total memory (mean × count), so hotspots appear last. Pass
        ``only_keys`` to restrict to a subset.
        """
        stats = {k: ((v, "B"), n, s) for k, (v, n, s) in self._memory_stats.items()}

        if only_keys is not None:
            stats = {k: v for k, v in stats.items() if k in only_keys}

        return pprint_stats_map(stats, self._CUTOFFS_AND_UNITS)
