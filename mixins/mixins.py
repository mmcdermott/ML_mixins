import functools, pickle, random, time

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

import random, numpy as np

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

def doublewrap(f):
    '''
    a decorator decorator, allowing the decorator to be used as:
    @decorator(with, arguments, and=kwargs)
    or
    @decorator
    '''
    @functools.wraps(f)
    def new_dec(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # actual decorated function
            return f(args[0])
        else:
            # decorator arguments
            return lambda realf: f(realf, *args, **kwargs)

    return new_dec

try:
    from tqdm.auto import tqdm

    class TQDMableMixin():
        _SKIP_TQDM_IF_LE = 3

        def __init__(self, *args, **kwargs):
            if not hasattr(self, 'tqdm'): self.tqdm = kwargs.get('tqdm', tqdm)

        def _tqdm(self, rng, **kwargs):
            if not hasattr(self, 'tqdm'): self.tqdm = tqdm

            if self.tqdm is None: return rng

            try: N = len(rng)
            except: return rng

            if N <= self._SKIP_TQDM_IF_LE: return rng

            return tqdm(rng, **kwargs)
except ImportError as e: pass

class SeedableMixin():
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

class SaveableMixin():
    _DEL_BEFORE_SAVING_ATTRS = []

    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'do_overwrite'): self.do_overwrite = kwargs.get('do_overwrite', False)

    @staticmethod
    def _load(filepath: Path, **add_kwargs) -> None:
        assert filepath.is_file(), f"Missing filepath {filepath}!"
        with open(fp, mode='rb') as f: obj = pickle.load(f)

        for a, v in add_kwargs.items(): setattr(obj, a, v)
        obj._post_load(add_kwargs)

        return obj

    def _post_load(self, load_add_kwargs: dict) -> None:
        # Overwrite this in the base class if desired.
        return

    def _save(self, filepath: Path, do_overwrite: Optional[bool] = False) -> None:
        if not hasattr(self, 'do_overwrite'): self.do_overwrite = False
        if not (self.do_overwrite or do_overwrite):
            assert not filepath.exists(), f"Filepath {filepath} already exists!"

        skipped_attrs = {}
        for attr in self._DEL_BEFORE_SAVING_ATTRS:
            if hasattr(self, attr): skipped_attrs[attr] = self.__dict__.pop(attr)

        with open(fp, mode='wb') as f: pickle.dump(self, f)

        for attr, val in skipped_attrs.items(): setattr(self, attr, val)

class TimeableMixin():
    _START_TIME = 'start'
    _END_TIME = 'end'

    def _register_start(self, key):
        if not hasattr(self, '_timings'): self._timings = defaultdict(list)

        self._timings[key].append({self._START_TIME: time.time()})

    def _register_end(self, key):
        assert hasattr(self, '_timings')
        assert key in self._timings and len(self._timings[key]) > 0
        assert self._timings[key][-1].get(self._END_TIME, None) is None
        self._timings[key][-1][self._END_TIME] = time.time()

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

class SwapcacheableMixin():
    def __init__(self, *args, **kwargs):
        if not hasattr(self, '_cache_size'): self._cache_size = kwargs.get('cache_size', 5)

    def _init_attrs(self):
        if not hasattr(self, '_cache'): self._cache = {'keys': [], 'values': []}
        if not hasattr(self, '_cache_size'): self._cache_size = 5
        if not hasattr(self, '_front_attrs'): self._front_attrs = []
        if not hasattr(self, '_front_cache_key'): self._front_cache_key = None
        if not hasattr(self, '_front_cache_idx'): self._front_cache_idx = None

    def _set_swapcache_key(self, key: Any):
        self._init_attrs()
        if key == self._front_cache_key: return

        seen_key = self._swapcache_has_key(key)
        if seen_key:
            idx = next(i for i, (k, t) in enumerate(self._seen_parameters) if k == key)
        else:
            self._cache['keys'].append((key, time.time()))
            self._cache['values'].append({})

            self._cache['keys'] = self._cache['keys'][-self._cache_size:]
            self._cache['values'] = self._cache['values'][-self._cache_size:]

            idx = -1

        # Clear out the old front-and-center attributes
        for attr in self._front_attrs: delattr(self, attr)

        self._front_cache_key = key
        self._front_cache_idx = idx

        self._update_front_attrs()

    def _swapcache_has_key(self, key: Any) -> bool:
        self._init_attrs()
        return any(k == key for k, t in self._cache['keys'])

    def _swap_to_key(self, key: Any) -> None:
        self._init_attrs()
        assert self._swapcache_has_key(key)
        self._set_swapcache_key(key)

    def _update_front_attrs(self):
        self._init_attrs()
        # Set the new front-and-center attributes
        for key, val in self._cache['values'][self._front_cache_idx].items(): setattr(self, key, val)

    def _update_swapcache_key_and_swap(self, key: Any, values_dict: dict):
        self._init_attrs()
        assert key is not None

        self._set_swapcache_key(key)
        self._cache['values'][self._front_cache_idx].update(values_dict)
        self._update_front_attrs()

    def _update_current_swapcache_key(self, values_dict: dict):
        self._init_attrs()
        self._update_swapcache_key_and_swap(self._front_cache_key, values_dict)
