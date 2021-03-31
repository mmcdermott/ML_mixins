import pickle, random
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from pytorch_lightning import seed_everything
except ImportError as e:
    import random, numpy as np

    def seed_everything(seed: Optional[int] = None, try_import_torch: Optional[bool] = True) -> int:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min

        try:
            if seed is None: seed = os.environ.get("PL_GLOBAL_SEED")
            seed = int(seed)
        except (TypeError, ValueError):
            seed = np.random.randint(min_seed_value, max_seed_value)
            print(f"No correct seed found, seed set to {seed}")

        assert (min_seed_value <= seed <= max_seed_value)

        random.seed(seed)
        np.random.seed(seed)
        if try_import_torch:
            try:
                import torch
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            except ImportError: pass

        return seed


class SeedableMixin():
    def _last_seed(self, name: str):
        for idx, (s, n, time) in enumerate(self._past_seeds[::-1]):
            if n == name:
                idx = len(self._past_seeds) - 1 - idx
                return idx, s

        print(f"Failed to find seed with name {name}!")
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
        if not hasattr(self, '_timings'): self._timings = {}

        if key not in self.timings: self.timings[key] = []
        self.timings[key].append({self._START_TIME: time.time()})

    def _register_end(self, key):
        assert hasattr(self, '_timings')
        assert key in self.timings and len(self.timings[key]) > 0
        assert self.timings[key][-1].get(self._END_TIME, None) is None
        self.timings[key][-1][self._END_TIME] = time.time()

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
        if key == self._front_cache_key: return

        seen_key = any(k == key for k, t in self._cache['keys'])
        if seen_key:
            idx = next(i, for i, (k, t) in enumerate(self._seen_parameters) if k == key)
        else:
            self._cache['keys'].append((key, time.time()))
            self._cache['values'].append({})

            self._cache['keys'] = self._cache['keys'][-self.cache_size:]
            self._cache['values'] = self._cache['values'][-self.cache_size:]

            idx = -1

        # Clear out the old front-and-center attributes
        for attr in self._front_attrs: delattr(self, attr)

        self._front_cache_key = key
        self._front_cache_idx = idx

        self._update_front_attrs()

    def _update_front_attrs(self):
        # Set the new front-and-center attributes
        for key, val in self._cache['values'][self._front_cache_idx].items(): setattr(self, key, val)

    def _update_swapcache_key_and_swap(key: Any, values_dict: dict):
        assert key is not None

        self._set_swapcache_key(key)
        self._cache['values'][self._front_cache_idx].update(values_dict)
        self._update_front_attrs()

    def _update_current_swapcache_key(values_dict: dict):
        self._update_swapcache_key_and_swap(self._front_cache_key, values_dict)
