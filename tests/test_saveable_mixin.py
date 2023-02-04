import sys
sys.path.append('..')

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from mixins import SaveableMixin

class Derived(SaveableMixin):
    _PICKLER = 'pickle'

    def __init__(self, a: int = -1, b: str = 'unset', **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and (self.a == other.a) and (self.b == other.b)

class DillDerived(SaveableMixin):
    _PICKLER = 'dill'

    def __init__(self, a: int = -1, b: str = 'unset', **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and (self.a == other.a) and (self.b == other.b)

class BadDerived(SaveableMixin):
    _PICKLER = 'not_supported'

    def __init__(self, a: int = -1, b: str = 'unset', **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __eq__(self, other: Any) -> bool:
        return type(self) == type(other) and (self.a == other.a) and (self.b == other.b)

class TestSaveableMixin(unittest.TestCase):
    def test_saveable_mixin(self):
        T = Derived(a=2, b='hi')

        with TemporaryDirectory() as d:
            save_path = Path(d) / 'save.pkl'
            T._save(save_path)

            with self.assertRaises(FileExistsError):
                new_t = Derived(a=3, b='bar')
                new_t._save(save_path)

            got_T = Derived._load(save_path)
            self.assertEqual(T, got_T)

            bad_T = BadDerived(a=2, b='hi')
            with self.assertRaises(NotImplementedError): bad_T._save(Path(d) / 'no_save.pkl')

            # This should error as that pickler isn't supported.
            with self.assertRaises(FileNotFoundError): got_T = Derived._load(Path(d) / 'no_save.pkl')
            with self.assertRaises(IsADirectoryError): got_T = Derived._load(Path(d))

            # This should error as dill isn't installed.
            with self.assertRaises(ImportError): bad_T = DillDerived(a=3, b='baz')
            T._PICKLER = 'dill'
            with self.assertRaises(ImportError): T._save(Path(d) / 'no_save.pkl')
            Derived._PICKLER = 'dill'
            with self.assertRaises(ImportError): got_T = Derived._load(save_path)

if __name__ == '__main__': unittest.main()
