import numpy as np

from mixins import MemTrackableMixin


class MemTrackableDerived(MemTrackableMixin):
    def __init__(self):
        self.foo = "foo"
        # Doesn't call super().__init__()! Should still work in this case.

    def uses_contextlib(self, mem_size_64b: int = 800000):
        with self._track_memory_as("using_contextlib"):
            np.ones((mem_size_64b,), dtype=np.float64)

    @MemTrackableMixin.TrackMemoryAs(key="decorated")
    def decorated_takes_mem(self, mem_size_64b: int = 800000):
        np.ones((mem_size_64b,), dtype=np.float64)

    @MemTrackableMixin.TrackMemoryAs
    def decorated_takes_mem_auto_key(self, mem_size_64b: int = 800000):
        np.ones((mem_size_64b,), dtype=np.float64)


def test_constructs():
    MemTrackableMixin()
    MemTrackableDerived()


def test_errors_if_not_initialized():
    M = MemTrackableDerived()
    try:
        M._peak_mem_for("foo")
        raise AssertionError("Should have raised an exception!")
    except AttributeError as e:
        assert "self._mem_stats should exist!" in str(e)

    M.uses_contextlib(mem_size_64b=8000000)  # 64 MB
    try:
        M._peak_mem_for("wrong_key")
        raise AssertionError("Should have raised an exception!")
    except AttributeError as e:
        assert "wrong_key should exist in self._mem_stats!" in str(e)


def test_benchmark_timing(benchmark):
    M = MemTrackableDerived()

    benchmark(M.decorated_takes_mem, 8000)


def test_context_manager():
    M = MemTrackableDerived()

    M.uses_contextlib(mem_size_64b=8000000)  # 64 MB

    mem_used = M._peak_mem_for("using_contextlib")[-1]
    np.testing.assert_almost_equal(mem_used, 64 * 1000000, decimal=1)

    M.uses_contextlib(mem_size_64b=80000)  # 0.64 MB

    mem_used = M._peak_mem_for("using_contextlib")[-1]
    np.testing.assert_almost_equal(mem_used, 64 * 10000, decimal=1)


def test_decorators_and_profiling():
    M = MemTrackableDerived()
    M.decorated_takes_mem(mem_size_64b=16000)

    mem_used = M._peak_mem_for("decorated")[-1]
    np.testing.assert_almost_equal(mem_used, 64 * 2000, decimal=1)

    M.decorated_takes_mem_auto_key(mem_size_64b=40000)
    mem_used = M._peak_mem_for("decorated_takes_mem_auto_key")[-1]
    np.testing.assert_almost_equal(mem_used, 64 * 5000, decimal=1)

    M.decorated_takes_mem(mem_size_64b=8000)
    stats = M._memory_stats

    assert {"decorated", "decorated_takes_mem_auto_key"} == set(stats.keys())
    np.testing.assert_almost_equal(64 * 1500, stats["decorated"][0], decimal=1)
    assert 2 == stats["decorated"][1]
    np.testing.assert_almost_equal(0.5 * 64 * 1000, stats["decorated"][2], decimal=1)
    np.testing.assert_almost_equal(64 * 5000, stats["decorated_takes_mem_auto_key"][0], decimal=1)
    assert 1 == stats["decorated_takes_mem_auto_key"][1]
    assert stats["decorated_takes_mem_auto_key"][2] is None

    got_str = M._profile_memory_usages()
    want_str = "decorated:                    96.0 ± 32.0 kB (x2)\ndecorated_takes_mem_auto_key: 320.0 kB"
    assert want_str == got_str, f"Want:\n{want_str}\nGot:\n{got_str}"

    got_str = M._profile_memory_usages(only_keys=["decorated_takes_mem_auto_key"])
    want_str = "decorated_takes_mem_auto_key: 320.0 kB"
    assert want_str == got_str, f"Want:\n{want_str}\nGot:\n{got_str}"
