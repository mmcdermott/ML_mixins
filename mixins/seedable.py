from __future__ import annotations

import functools, random, numpy as np

from datetime import datetime
from typing import Optional

from .utils import doublewrap

def seed_everything(seed: Optional[int] = None, try_import_torch: Optional[bool] = True) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None: seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = np.random.randint(min_seed_value, max_seed_value)

    assert (min_seed_value <= seed <= max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    if try_import_torch:
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ModuleNotFoundError: pass

    return seed

class SeedableMixin():
    def __init__(self, *args, **kwargs):
        self._past_seeds = kwargs.get('_past_seeds', [])

    def _last_seed(self, key: str):
        for idx, (s, k, time) in enumerate(self._past_seeds[::-1]):
            if k == key:
                idx = len(self._past_seeds) - 1 - idx
                return idx, s

        return -1, None

    def _seed(self, seed: Optional[int] = None, key: Optional[str] = None):
        if seed is None: seed = random.randint(0, int(1e8))
        if key is None: key = ''
        time = str(datetime.now())

        self.seed = seed
        if hasattr(self, '_past_seeds'): self._past_seeds.append((self.seed, key, time))
        else: self._past_seeds = [(self.seed, key, time)]

        seed_everything(seed)
        return seed

    @staticmethod
    @doublewrap
    def WithSeed(fn, key: Optional[str] = None):
        if key is None: key = fn.__name__
        @functools.wraps(fn)
        def wrapper_seeding(self, *args, seed: Optional[int] = None, **kwargs):
            self._seed(seed=seed, key=key)
            return fn(self, *args, **kwargs)
        return wrapper_seeding

