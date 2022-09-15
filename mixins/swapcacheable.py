from __future__ import annotations

import time
from typing import Hashable

class SwapcacheableMixin():
    def __init__(self, *args, **kwargs):
        self._cache_size = kwargs.get('cache_size', 5)

    def _init_attrs(self):
        if not hasattr(self, '_cache'): self._cache = {'keys': [], 'values': []}
        if not hasattr(self, '_cache_size'): self._cache_size = 5
        if not hasattr(self, '_front_attrs'): self._front_attrs = []
        if not hasattr(self, '_front_cache_key'): self._front_cache_key = None
        if not hasattr(self, '_front_cache_idx'): self._front_cache_idx = None

    def _set_swapcache_key(self, key: Hashable) -> None:
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

    def _swapcache_has_key(self, key: Hashable) -> bool:
        self._init_attrs()
        return any(k == key for k, t in self._cache['keys'])

    def _swap_to_key(self, key: Hashable) -> None:
        self._init_attrs()
        assert self._swapcache_has_key(key)
        self._set_swapcache_key(key)

    def _update_front_attrs(self):
        self._init_attrs()
        # Set the new front-and-center attributes
        for key, val in self._cache['values'][self._front_cache_idx].items(): setattr(self, key, val)

    def _update_swapcache_key_and_swap(self, key: Hashable, values_dict: dict):
        self._init_attrs()
        assert key is not None

        self._set_swapcache_key(key)
        self._cache['values'][self._front_cache_idx].update(values_dict)
        self._update_front_attrs()

    def _update_current_swapcache_key(self, values_dict: dict):
        self._init_attrs()
        self._update_swapcache_key_and_swap(self._front_cache_key, values_dict)
