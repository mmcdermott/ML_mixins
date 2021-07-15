import sys
sys.path.append('..')

import unittest

from mixins.mixins import *

try:
    import torch

    class TestTensorableMixin(unittest.TestCase):
        def test_constructs(self):
            T = TensorableMixin()

except ImportError: pass

if __name__ == '__main__': unittest.main()

