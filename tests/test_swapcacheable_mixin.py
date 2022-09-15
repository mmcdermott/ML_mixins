import sys
sys.path.append('..')

import unittest

from mixins import SwapcacheableMixin

class TestSwapcacheableMixin(unittest.TestCase):
    def test_constructs(self):
        T = SwapcacheableMixin()

if __name__ == '__main__': unittest.main()

