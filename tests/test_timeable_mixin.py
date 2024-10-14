import time

import numpy as np

from mixins import TimeableMixin


class TimeableDerived(TimeableMixin):
    def __init__(self):
        self.foo = "foo"
        # Doesn't call super().__init__()! Should still work in this case.

    def uses_contextlib(self, num_seconds: int = 5):
        with self._time_as("using_contextlib"):
            time.sleep(num_seconds)

    @TimeableMixin.TimeAs(key="decorated")
    def decorated_takes_time(self, num_seconds: int = 10):
        time.sleep(num_seconds)

    @TimeableMixin.TimeAs
    def decorated_takes_time_auto_key(self, num_seconds: int = 10):
        time.sleep(num_seconds)


def test_constructs():
    TimeableMixin()
    TimeableDerived()


def test_responds_to_methods():
    T = TimeableMixin()

    T._register_start("key")
    T._register_end("key")

    T._times_for("key")

    T._register_start("key")
    T._time_so_far("key")
    T._register_end("key")


def test_benchmark_timing(benchmark):
    T = TimeableDerived()

    benchmark(T.decorated_takes_time, 0.00001)


def test_pprint_num_unit():
    assert (5, "μs") == TimeableMixin._get_pprint_num_unit(5 * 1e-6)

    class Derived(TimeableMixin):
        _CUTOFFS_AND_UNITS = [(10, "foo"), (2, "bar"), (None, "biz")]

    assert (3, "biz") == Derived._get_pprint_num_unit(3, "biz")
    assert (3, "foo") == Derived._get_pprint_num_unit(3 / 20, "biz")
    assert (1.2, "biz") == Derived._get_pprint_num_unit(2.4 * 10, "foo")

    try:
        Derived._get_pprint_num_unit(1, "WRONG")
        raise AssertionError("Should have raised an exception")
    except LookupError:
        pass
    except Exception as e:
        raise AssertionError(f"Raised the wrong exception: {e}")


def test_context_manager():
    T = TimeableDerived()

    T.uses_contextlib(num_seconds=1)

    duration = T._times_for("using_contextlib")[-1]
    np.testing.assert_almost_equal(duration, 1, decimal=1)


def test_times_and_profiling():
    T = TimeableDerived()
    T.decorated_takes_time(num_seconds=2)

    duration = T._times_for("decorated")[-1]
    np.testing.assert_almost_equal(duration, 2, decimal=1)

    T.decorated_takes_time_auto_key(num_seconds=2)
    duration = T._times_for("decorated_takes_time_auto_key")[-1]
    np.testing.assert_almost_equal(duration, 2, decimal=1)

    T.decorated_takes_time(num_seconds=1)
    stats = T._duration_stats

    assert {"decorated", "decorated_takes_time_auto_key"} == set(stats.keys())
    np.testing.assert_almost_equal(1.5, stats["decorated"][0], decimal=1)
    assert 2 == stats["decorated"][1]
    np.testing.assert_almost_equal(0.5, stats["decorated"][2], decimal=1)
    np.testing.assert_almost_equal(2, stats["decorated_takes_time_auto_key"][0], decimal=1)
    assert 1 == stats["decorated_takes_time_auto_key"][1]
    assert 0 == stats["decorated_takes_time_auto_key"][2]

    got_str = T._profile_durations()
    want_str = "decorated_takes_time_auto_key: 2.0 sec\ndecorated:                     1.5 ± 0.5 sec (x2)"
    assert want_str == got_str, f"Want:\n{want_str}\nGot:\n{got_str}"

    got_str = T._profile_durations(only_keys=["decorated_takes_time_auto_key"])
    want_str = "decorated_takes_time_auto_key: 2.0 sec"
    assert want_str == got_str, f"Want:\n{want_str}\nGot:\n{got_str}"
