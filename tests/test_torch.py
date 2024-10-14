import random

import numpy as np

from mixins.seedable import seed_everything

try:
    import torch
except (ImportError, ModuleNotFoundError):
    raise ImportError("This test requires torch to run.")


def test_benchmark_seed_everything_torch(benchmark):
    benchmark(seed_everything, seed_engines={"torch"})


def test_seed_everything():
    seed_everything(1, seed_engines={"torch"})

    rand_1_1 = random.randint(0, 100000000)
    np_rand_1_1 = np.random.randint(0, 100000000)
    torch_rand_1_1 = torch.randint(0, 100000000, (1,)).item()
    rand_2_1 = random.randint(0, 100000000)
    np_rand_2_1 = np.random.randint(0, 100000000)
    torch_rand_2_1 = torch.randint(0, 100000000, (1,)).item()

    seed_everything(1, seed_engines={"torch"})

    rand_1_2 = random.randint(0, 100000000)
    np_rand_1_2 = np.random.randint(0, 100000000)
    torch_rand_1_2 = torch.randint(0, 100000000, (1,)).item()
    rand_2_2 = random.randint(0, 100000000)
    np_rand_2_2 = np.random.randint(0, 100000000)
    torch_rand_2_2 = torch.randint(0, 100000000, (1,)).item()

    assert rand_1_1 != rand_1_2
    assert rand_1_1 != rand_2_1
    assert rand_2_1 != rand_2_2

    assert np_rand_1_1 != np_rand_1_2
    assert np_rand_1_1 != np_rand_2_1
    assert np_rand_2_1 != np_rand_2_2

    assert torch_rand_1_1 == torch_rand_1_2
    assert torch_rand_1_1 != torch_rand_2_1
    assert torch_rand_2_1 == torch_rand_2_2
