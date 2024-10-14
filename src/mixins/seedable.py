from __future__ import annotations

import functools
import os
import random
from datetime import datetime

import numpy as np

_SEED_FUNCTIONS = {
    "numpy": np.random.seed,
    "random": random.seed,
}

try:
    import torch

    def seed_torch(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    _SEED_FUNCTIONS["torch"] = seed_torch
except ModuleNotFoundError:
    pass


from .utils import doublewrap


def seed_everything(seed: int | None = None, seed_engines: set[str] | None = None) -> int:
    """A simple helper function to seed everything that needs to be seeded.

    Args:
        seed: The seed to use. If None, a random seed is chosen.

    Returns:
        The seed that was used.

    Examples:
        >>> random.seed(0)
        >>> np.random.seed(0)
        >>> random.randint(0, 10)
        6
        >>> random.randint(0, 10)
        6
        >>> np.random.randint(0, 10)
        5
        >>> np.random.randint(0, 10)
        0
        >>> seed_everything(0)
        0
        >>> random.randint(0, 10)
        6
        >>> random.randint(0, 10)
        6
        >>> np.random.randint(0, 10)
        5
        >>> np.random.randint(0, 10)
        0
    """

    if seed_engines is None:
        seed_engines = set(_SEED_FUNCTIONS.keys())

    if seed is None:
        if "PL_GLOBAL_SEED" in os.environ:
            seed = int(os.environ["PL_GLOBAL_SEED"])
        else:
            max_seed_value = np.iinfo(np.uint32).max
            min_seed_value = np.iinfo(np.uint32).min
            seed = np.random.randint(min_seed_value, max_seed_value)

    for s in seed_engines:
        _SEED_FUNCTIONS[s](seed)

    return seed


class SeedableMixin:
    """This class provides easy utilities to reliably seed stochastic processes.

    This seeding can be used to ensure reproducibility in experiments, both in individual examples with an
    integral seed or in a stochastic process both at a per-event level and at a whole process level by seeding
    with `None`, in which case a new seed is chosen for each event in the process based on the prior seed and
    stored.
    """

    def __init__(self, *args, **kwargs):
        self._past_seeds = kwargs.get("_past_seeds", [])
        self._seed_engines = kwargs.get("_seed_engines", set(_SEED_FUNCTIONS.keys()))

    def _last_seed(self, key: str) -> tuple[int, int | None]:
        """This returns the most recently used seed with a given key.

        Args:
            key: The key to search for.

        Returns:
            The index of the most recent seed with a given key in the list of past seeds and the seed itself.

        Examples:
            >>> M = SeedableMixin()
            >>> _ = M._seed(0, "foo")
            >>> _ = M._seed(2, "bar")
            >>> _ = M._seed(4, "foo")
            >>> _ = M._seed(6, "baz")
            >>> M._last_seed("foo")
            (2, 4)
            >>> M._last_seed("bar")
            (1, 2)
            >>> M._last_seed("baz")
            (3, 6)
        """
        for idx, (s, k, time) in enumerate(self._past_seeds[::-1]):
            if k == key:
                idx = len(self._past_seeds) - 1 - idx
                return idx, s

        return -1, None

    def _seed(self, seed: int | None = None, key: str | None = None) -> int:
        """This seeds the random number generators.

        Args:
            seed: The seed to use. If None, a new seed is chosen.
            key: The key to associate with this seed.

        Returns:
            The seed that was used.

        Examples:
            >>> M = SeedableMixin()
            >>> M._seed(0, "foo")
            0
            >>> M._seed(2, "bar")
            2
            >>> M._seed(4, "foo")
            4

        Note that by virtue of the fact that we've already seeded `M`, future seeds are deterministic (though
        they are still pseudo-random, as they are simply random integers drawn from the current random
        distribution, which in this test was seeded at 4 immediately prior to this call).
            >>> M._seed()
            31681838

        Past seeds and keys are stored in the `_past_seeds` attribute, which is created if the object does not
        have it at the start.
            >>> M = SeedableMixin()
            >>> del M._past_seeds
            >>> M._seed(0, "foo")
            0
            >>> M._seed(2, "bar")
            2
            >>> M._seed(4, "foo")
            4
            >>> M._seed()
            31681838
            >>> M._past_seeds
            [(0, 'foo', ...), (2, 'bar', ...), (4, 'foo', ...), (31681838, '', ...)]
        """
        if seed is None:
            seed = random.randint(0, int(1e8))
        if key is None:
            key = ""
        time = str(datetime.now())

        self.seed = seed
        if hasattr(self, "_past_seeds"):
            self._past_seeds.append((self.seed, key, time))
        else:
            self._past_seeds = [(self.seed, key, time)]

        seed_everything(seed, getattr(self, "_seed_engines", None))
        return seed

    @staticmethod
    @doublewrap
    def WithSeed(fn, key: str | None = None) -> callable:
        """This function is a decorator that returns a function that also takes a seed which seeds the RNG.

        This decorator can either be called with a `key` argument or without arguments. In the latter case,
        the decorator is used like this:

        ```
        @SeedableMixin.WithSeed
        def func(...):
            ...
        ```

        In this case, the name of the function is used as the key to the associated seed call. If a key is
        provided, the decorator is used like this:

        ```
        @SeedableMixin.WithSeed(key="foo")
        def func(...):
            ...
        ```

        In this case, the key is used as the key to the associated seed call. This is useful when the function
        name is not the desired seed.

        Args:
            fn: The function to wrap. This argument _does not need to be provided_ if a key is used; instead
                the `doublewrap` decorator is used to allow the key to be passed as a keyword argument to a
                meta-function that returns the true decorator applied to the target function.
            key: The key to use for the seed. If None, the function name is used.

        Returns:
            A function that takes all the input arguments of the wrapped function and a seed keyword argument.
            If the seed is not provided, a new seed is chosen. The seed is used to seed the RNG before calling
            the wrapped function, under the provided key.

        Note that if the function being wrapped explicitly takes a seed argument, this decorator will not
        work, and the failure will not necessarily be graceful.

        Examples:
        """
        if key is None:
            key = fn.__name__

        @functools.wraps(fn)
        def wrapper_seeding(self, *args, seed: int | None = None, **kwargs):
            self._seed(seed=seed, key=key)
            return fn(self, *args, **kwargs)

        return wrapper_seeding
