import unittest

from mixins import TQDMableMixin


class TestTQDMableMixin(unittest.TestCase):
    def test_constructs(self):
        TQDMableMixin()


if __name__ == "__main__":
    unittest.main()
