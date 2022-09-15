from __future__ import annotations

import functools, inspect, pickle
from copy import deepcopy
from pathlib import Path
from typing import Optional

from .utils import doublewrap

class DebuggableMixin:
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
                T = inspect.trace()
                for t in T:
                    if t[3] == fn.__name__: break

                new_vars = deepcopy(t[0].f_locals)
                if store_global:
                    __builtins__["_DEBUGGER_VARS"] = new_vars
                if filepath:
                    with open(filepath, mode='wb') as f:
                        pickle.dump(new_vars, f)
                raise
        return debugging_wrapper
