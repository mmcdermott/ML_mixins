import os
import random

import numpy as np

from mixins import SeedableMixin
from mixins.seedable import seed_everything

try:
    pass

    raise ImportError("This test requires torch not to be installed to run.")
except (ImportError, ModuleNotFoundError):
    pass


class SeedableDerived(SeedableMixin):
    def __init__(self):
        self.foo = "foo"
        # Doesn't call super().__init__()! Should still work in this case.

    def gen_random_num(self):
        # This is purely random
        return (random.random(), np.random.rand())

    @SeedableMixin.WithSeed(key="decorated")
    def decorated_gen_random_num(self):
        return (random.random(), np.random.rand())

    @SeedableMixin.WithSeed
    def decorated_auto_key(self):
        return random.random()


def test_benchmark_seed_everything(benchmark):
    benchmark(seed_everything)


def test_benchmark_seed_everything_with_seed(benchmark):
    benchmark(seed_everything, 1)


def test_benchmark_seed_everything_with_env(benchmark):
    os.environ["PL_GLOBAL_SEED"] = "1"
    benchmark(seed_everything)


def test_seed_everything():
    os.environ["PL_GLOBAL_SEED"] = "1"
    seed_everything()

    rand_1 = random.randint(0, 100000000)
    np_rand_1 = np.random.randint(0, 100000000)
    rand_2 = random.randint(0, 100000000)
    np_rand_2 = np.random.randint(0, 100000000)

    seed_everything(1)
    rand_1_1 = random.randint(0, 100000000)
    np_rand_1_1 = np.random.randint(0, 100000000)
    rand_2_1 = random.randint(0, 100000000)
    np_rand_2_1 = np.random.randint(0, 100000000)

    seed_everything(1, seed_engines={"random"})
    rand_1_2 = random.randint(0, 100000000)
    np_rand_1_2 = np.random.randint(0, 100000000)
    rand_2_2 = random.randint(0, 100000000)
    np_rand_2_2 = np.random.randint(0, 100000000)

    seed_everything(1, seed_engines={"numpy"})
    rand_1_3 = random.randint(0, 100000000)
    np_rand_1_3 = np.random.randint(0, 100000000)
    rand_2_3 = random.randint(0, 100000000)
    np_rand_2_3 = np.random.randint(0, 100000000)

    assert rand_1 == rand_1_1
    assert rand_1 == rand_1_2
    assert rand_1 != rand_1_3
    assert rand_1 != rand_2

    assert np_rand_1 == np_rand_1_1
    assert np_rand_1 == np_rand_1_3
    assert np_rand_1 != np_rand_1_2
    assert np_rand_1 != np_rand_2

    assert rand_2 == rand_2_1
    assert rand_2 == rand_2_2
    assert rand_2 != rand_2_3

    assert np_rand_2 == np_rand_2_1
    assert np_rand_2 == np_rand_2_3
    assert np_rand_2 != np_rand_2_2


def test_constructs():
    SeedableMixin()
    SeedableDerived()


def test_benchmark_seeding(benchmark):
    T = SeedableDerived()

    benchmark(T._seed)


def test_responds_to_methods():
    T = SeedableMixin()

    T._seed()
    T._last_seed("foo")

    T = SeedableDerived()
    T._seed()
    T._last_seed("foo")


# Note: `test_seeding_freezes_randomness`, `test_decorated_seeding_freezes_randomness`, and
# `test_seeds_follow_consistent_sequence` previously lived here; they have moved into doctests on
# `SeedableMixin`, `SeedableMixin.WithSeed`, and `README.md` where they double as documentation.


def test_get_last_seed():
    T = SeedableDerived()

    key = "key"
    non_key = "not_key"

    seed_key_early = 1
    seed_key_late = 1
    seed_non_key = 2

    T._seed()

    idx, seed = T._last_seed(key)
    assert idx == -1
    assert seed is None

    T._seed(seed_key_early, key)
    T._seed()
    T._seed(seed_non_key, non_key)
    T._seed(seed_key_late, key)
    T._seed(seed_non_key, non_key)

    idx, seed = T._last_seed(key)
    assert idx == 4
    assert seed == seed_key_late
