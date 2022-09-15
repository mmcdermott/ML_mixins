import sys, time, numpy as np
sys.path.append('..')

import unittest

from mixins import TimeableMixin

class TimeableDerived(TimeableMixin):
    def __init__(self):
        self.foo = 'foo'
        # Doesn't call super().__init__()! Should still work in this case.

    def uses_contextlib(self, num_seconds: int = 5):
        with self._time_as('using_contextlib'):
            time.sleep(num_seconds)

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

        T._times_for('key')

        T._register_start('key')
        T._time_so_far('key')
        T._register_end('key')

    def test_context_manager(self):
        T = TimeableDerived()

        T.uses_contextlib()

        duration = T._times_for('using_contextlib')[-1]
        np.testing.assert_almost_equal(duration, 5, decimal=1)

    def test_times(self):
        T = TimeableDerived()
        T.decorated_takes_time(num_seconds=10)

        duration = T._times_for('decorated')[-1]
        np.testing.assert_almost_equal(duration, 10, decimal=1)

        T.decorated_takes_time_auto_key(num_seconds=10)
        duration = T._times_for('decorated_takes_time_auto_key')[-1]
        np.testing.assert_almost_equal(duration, 10, decimal=1)

if __name__ == '__main__': unittest.main()
