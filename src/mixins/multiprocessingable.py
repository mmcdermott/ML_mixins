from __future__ import annotations

from collections.abc import Callable, Sequence
from multiprocessing import Pool


class MultiprocessingMixin:
    def __init__(self, *args, multiprocessing_pool_size: int | None = None, **kwargs):
        self.multiprocessing_pool_size = multiprocessing_pool_size

    @property
    def _multiprocessing_pool_size(self):
        if hasattr(self, "multiprocessing_pool_size"):
            return self.multiprocessing_pool_size
        else:
            return None

    @property
    def _use_multiprocessing(self):
        return self._multiprocessing_pool_size is not None and self._multiprocessing_pool_size > 1

    def _map(self, fn: Callable, iterable: Sequence, tqdm: Callable | None = None, **tqdm_kwargs) -> Sequence:
        if self._use_multiprocessing:
            with Pool(self._multiprocessing_pool_size) as p:
                if tqdm is None:
                    return p.map(fn, iterable)
                else:
                    return list(tqdm(p.imap(fn, iterable), **tqdm_kwargs))
        elif tqdm is None:
            return [fn(x) for x in iterable]
        else:
            return [fn(x) for x in tqdm(iterable, **tqdm_kwargs)]
