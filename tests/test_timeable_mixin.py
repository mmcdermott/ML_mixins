import sys, time, numpy as np
sys.path.append('..')

import unittest

from mixins.mixins import *

class TimeableDerived(TimeableMixin):
    def __init__(self):
        self.foo = 'foo'
        # Doesn't call super().__init__()! Should still work in this case.

    @TimeableMixin.TimeAs(key="decorated")
    def decorated_takes_time(self, num_seconds: int = 10):
        time.sleep(num_seconds)

    @TimeableMixin.TimeAs
    def decorated_takes_time_auto_key(self, num_seconds: int = 10):
        time.sleep(num_seconds)

class TestTimeableMixin(unittest.TestCase):
    def test_constructs(self):
        T = TimeableMixin()
        T = TimeableDerived()

    def test_responds_to_methods(self):
        T = TimeableMixin()

        T._register_start('key')
        T._register_end('key')

    def test_times(self):
        T = TimeableDerived()
        T.decorated_takes_time(num_seconds=10)

        tt = T._timings['decorated'][-1]
        duration = tt['end'] - tt['start']
        np.testing.assert_almost_equal(duration, 10, decimal=1)

        T.decorated_takes_time_auto_key(num_seconds=10)
        tt = T._timings['decorated_takes_time_auto_key'][-1]
        duration = tt['end'] - tt['start']
        np.testing.assert_almost_equal(duration, 10, decimal=1)

if __name__ == '__main__': unittest.main()
