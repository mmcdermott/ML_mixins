from __future__ import annotations

import functools, inspect, pickle, random, time, numpy as np

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
            self.tqdm = kwargs.get('tqdm', tqdm)

        def _tqdm(self, rng, **kwargs):
            if not hasattr(self, 'tqdm'): self.tqdm = tqdm

            if self.tqdm is None: return rng

            try: N = len(rng)
            except: return rng

            if N <= self._SKIP_TQDM_IF_LE: return rng

            return tqdm(rng, **kwargs)
except ImportError as e: pass

try:
    import torch

    class TensorableMixin():
        Tensorable_T = Union[np.ndarray, List[float], Tuple['Tensorable_T'], Dict[Any, 'Tensorable_T']]
        Tensor_T = Union[torch.Tensor, Tuple['Tensor_T'], Dict[Any, 'Tensor_T']]

        def __init__(self, *args, **kwargs):
            self.do_cuda = kwargs.get('do_cuda', torch.cuda.is_available)

        def _cuda(self, T: torch.Tensor, do_cuda: Optional[bool] = None):
            if do_cuda is None:
                do_cuda = self.do_cuda if hasattr(self, 'do_cuda') else torch.cuda.is_available

            return T.cuda() if do_cuda else T

        def _from_numpy(self, obj: np.ndarray) -> torch.Tensor:
            # I keep getting errors about "RuntimeError: expected scalar type Float but found Double"
            if obj.dtype == np.float64: obj = obj.astype(np.float32)
            return self._cuda(torch.from_numpy(obj))

        def _nested_to_tensor(self, obj: TensorableMixin.Tensorable_T) -> TensorableMixin.Tensor_T:
            if isinstance(obj, np.ndarray): return self._from_numpy(obj)
            elif isinstance(obj, list): return self._from_numpy(np.array(obj))
            elif isinstance(obj, dict): return {k: self._nested_to_tensor(v) for k, v in obj.items()}
            elif isinstance(obj, tuple): return tuple((self._nested_to_tensor(e) for e in obj))

            raise ValueError(f"Don't know how to convert {type(obj)} object {obj} to tensor!")

except ImportError as e: pass

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

class SaveableMixin():
    _DEL_BEFORE_SAVING_ATTRS = []

    def __init__(self, *args, **kwargs):
        self.do_overwrite = kwargs.get('do_overwrite', False)

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

class SwapcacheableMixin():
    def __init__(self, *args, **kwargs):
        self._cache_size = kwargs.get('cache_size', 5)

    def _init_attrs(self):
        if not hasattr(self, '_cache'): self._cache = {'keys': [], 'values': []}
        if not hasattr(self, '_cache_size'): self._cache_size = 5
        if not hasattr(self, '_front_attrs'): self._front_attrs = []
        if not hasattr(self, '_front_cache_key'): self._front_cache_key = None
        if not hasattr(self, '_front_cache_idx'): self._front_cache_idx = None

    def _set_swapcache_key(self, key: Any) -> None:
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

class DebuggerMixin:
    @property
    def _do_debug(self):
        if hasattr(self, 'do_debug'): return self.do_debug
        else: return False

    @staticmethod
    @doublewrap
    def CaptureErrorState(fn, store_global: Optional[bool] = None, filepath: Optional[Path] = None):
        if store_global is None: store_global = (filepath is None)

        @functools.wraps(fn)
        def debugging_wrapper(self, *args, seed: Optional[int] = None, **kwargs):
            if not self._do_debug: return fn(self, *args, **kwargs)
            
            try:
                return fn(self, *args, **kwargs)
            except Exception as e:
                new_vars = deepcopy(inspect.trace()[-1][0].f_locals)
                if store_global:
                    __builtins__["_DEBUGGER_VARS"] = new_vars
                if filepath:
                    with open(filepath, mode='wb') as f:
                        pickle.dump(new_vars, f)
                raise
        return debugging_wrapper
