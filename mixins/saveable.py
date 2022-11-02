from __future__ import annotations

import pickle

from pathlib import Path
from typing import Optional

class SaveableMixin():
    _DEL_BEFORE_SAVING_ATTRS = []

    def __init__(self, *args, **kwargs):
        self.do_overwrite = kwargs.get('do_overwrite', False)

    @staticmethod
    def _load(filepath: Path, **add_kwargs) -> None:
        assert filepath.is_file(), f"Missing filepath {filepath}!"
        with open(filepath, mode='rb') as f: obj = pickle.load(f)

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

        with open(filepath, mode='wb') as f: pickle.dump(self, f)

        for attr, val in skipped_attrs.items(): setattr(self, attr, val)
