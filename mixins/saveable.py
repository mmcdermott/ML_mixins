from __future__ import annotations

import pickle as pickle
try:
    import dill
    dill_imported = True
    dill_import_error = None
except ImportError as e:
    dill_import_error = e
    dill_imported = False

from pathlib import Path
from typing import Optional

class SaveableMixin():
    _DEL_BEFORE_SAVING_ATTRS = []
    # TODO(mmd): Make StrEnum upon conversion to python 3.11
    _PICKLER = 'dill' if dill_imported else 'pickle'

    def __init__(self, *args, **kwargs):
        self.do_overwrite = kwargs.get('do_overwrite', False)
        if self._PICKLER == 'dill' and not dill_imported: raise dill_import_error

    @classmethod
    def _load(cls, filepath: Path, **add_kwargs) -> None:
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} does not exist.")
        elif not filepath.is_file():
            raise IsADirectoryError(f"{filepath} is not a file.")

        with open(filepath, mode='rb') as f:
            match cls._PICKLER:
                case 'dill':
                    if not dill_imported: raise dill_import_error
                    obj = dill.load(f)
                case 'pickle': obj = pickle.load(f)
                case _:
                    raise NotImplementedError(f"{cls._PICKLER} not supported! Options: {'dill', 'pickle'}")

        for a, v in add_kwargs.items(): setattr(obj, a, v)
        obj._post_load(add_kwargs)

        return obj

    def _post_load(self, load_add_kwargs: dict) -> None:
        # Overwrite this in the base class if desired.
        return

    def _save(self, filepath: Path, do_overwrite: Optional[bool] = False) -> None:
        if not hasattr(self, 'do_overwrite'): self.do_overwrite = False
        if not (self.do_overwrite or do_overwrite):
            if filepath.exists():
                raise FileExistsError(f"Filepath {filepath} already exists!")

        skipped_attrs = {}
        for attr in self._DEL_BEFORE_SAVING_ATTRS:
            if hasattr(self, attr): skipped_attrs[attr] = self.__dict__.pop(attr)

        try:
            with open(filepath, mode='wb') as f:
                match self._PICKLER:
                    case 'dill':
                        if not dill_imported: raise dill_import_error
                        dill.dump(self, f)
                    case 'pickle': pickle.dump(self, f)
                    case _:
                        raise NotImplementedError(
                            f"{self._PICKLER} not supported! Options: {'dill', 'pickle'}"
                        )
        except:
            filepath.unlink()
            raise


        for attr, val in skipped_attrs.items(): setattr(self, attr, val)
