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

        T.uses_contextlib(num_seconds=1)

        duration = T._times_for('using_contextlib')[-1]
        np.testing.assert_almost_equal(duration, 1, decimal=1)

    def test_times_and_profiling(self):
        T = TimeableDerived()
        T.decorated_takes_time(num_seconds=2)

        duration = T._times_for('decorated')[-1]
        np.testing.assert_almost_equal(duration, 2, decimal=1)

        T.decorated_takes_time_auto_key(num_seconds=2)
        duration = T._times_for('decorated_takes_time_auto_key')[-1]
        np.testing.assert_almost_equal(duration, 2, decimal=1)

        T.decorated_takes_time(num_seconds=1)
        stats = T._duration_stats

        self.assertEqual({'decorated', 'decorated_takes_time_auto_key'}, set(stats.keys()))
        np.testing.assert_almost_equal(1.5, stats['decorated'][0], decimal=1)
        self.assertEqual(2, stats['decorated'][1])
        np.testing.assert_almost_equal(0.5, stats['decorated'][2], decimal=1)
        np.testing.assert_almost_equal(2, stats['decorated_takes_time_auto_key'][0], decimal=1)
        self.assertEqual(1, stats['decorated_takes_time_auto_key'][1])
        self.assertEqual(0, stats['decorated_takes_time_auto_key'][2])

        got_str = T._profile_durations()
        want_str = (
            "decorated_takes_time_auto_key: 2.0 sec\n"
            "decorated:                     1.5 ± 0.5 sec (x2)"
        )
        self.assertEqual(want_str, got_str, msg=f"Want:\n{want_str}\nGot:\n{got_str}")

if __name__ == '__main__': unittest.main()
