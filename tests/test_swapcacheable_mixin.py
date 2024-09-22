import unittest

from mixins import SwapcacheableMixin


class TestSwapcacheableMixin(unittest.TestCase):
    def test_constructs(self):
        SwapcacheableMixin()


if __name__ == "__main__":
    unittest.main()
