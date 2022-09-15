import sys
sys.path.append('..')

import unittest

from mixins import TQDMableMixin

class TestTQDMableMixin(unittest.TestCase):
    def test_constructs(self):
        T = TQDMableMixin()

if __name__ == '__main__': unittest.main()

