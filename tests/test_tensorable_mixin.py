import sys
sys.path.append('..')

import unittest

try:
    import torch
    from mixins import TensorableMixin

    class TestTensorableMixin(unittest.TestCase):
        def test_constructs(self):
            T = TensorableMixin()

except ImportError: pass

if __name__ == '__main__': unittest.main()

