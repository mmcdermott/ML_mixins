from __future__ import annotations

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
