import unittest

from mixins import TensorableMixin


class TestTensorableMixin(unittest.TestCase):
    def test_constructs(self):
        TensorableMixin()


if __name__ == "__main__":
    unittest.main()
