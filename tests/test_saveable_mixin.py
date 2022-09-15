import sys
sys.path.append('..')

import unittest

from mixins import SaveableMixin

class TestSaveableMixin(unittest.TestCase):
    def test_constructs(self):
        T = SaveableMixin()

if __name__ == '__main__': unittest.main()
