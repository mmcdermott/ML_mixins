from __future__ import annotations

import functools
import os
import random
import secrets
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
        seed: The seed to use. If ``None``, a fresh seed is chosen from OS entropy
            (``secrets.randbits(32)``) — **not** from the RNGs that this function is about to
            reseed. So two consecutive seedless calls produce unrelated seeds even if Python ``random``
            and ``numpy.random`` were just seeded to the same value. ``$PL_GLOBAL_SEED``, if set, takes
            precedence over the OS-entropy draw (for Lightning worker compatibility). Note that
            ``$PL_GLOBAL_SEED`` is **not** unset after being read, so every subsequent seedless call
            within the same process will pick up the same value — this matches Lightning's convention
            (the env var is written once by the parent process and inherited by workers) but can
            surprise callers who expect seedless calls to yield distinct seeds. In that case, pass
            ``seed=None`` only after explicitly clearing the env var, or always pass an explicit seed.
        seed_engines: The engines to seed. If ``None``, seeds every registered engine (``random``,
            ``numpy``, and ``torch`` if installed).

    Returns:
        The seed that was used.

    Examples:
        With an explicit seed, subsequent draws from ``random`` and ``numpy`` are reproducible:

        >>> seed_everything(0)
        0
        >>> random.randint(0, 10), np.random.randint(0, 10)
        (6, 5)
        >>> seed_everything(0)
        0
        >>> random.randint(0, 10), np.random.randint(0, 10)
        (6, 5)

        Without a seed, the contract is the opposite: the fallback is independent of the caller's
        current RNG state. Reseeding ``random`` and ``numpy`` to a fixed value between two seedless
        calls does **not** make them return the same seed — that's the point of drawing fresh
        entropy. (Collisions are possible at 1/2**32 odds but vanishingly rare; we assert that the
        two seeds are positive 32-bit ints rather than a specific equality to keep the doctest
        deterministic.)

        >>> seed_everything(0); a = seed_everything()
        0
        >>> seed_everything(0); b = seed_everything()
        0
        >>> 0 <= a < 2**32 and 0 <= b < 2**32
        True
        >>> isinstance(a, int) and isinstance(b, int)
        True

        ``seed_engines`` restricts which RNGs get reseeded — the engines *not* listed keep advancing from
        their current state:

        >>> _ = seed_everything(1)                           # reset both
        >>> a_py, a_np = random.random(), np.random.random()
        >>> _ = seed_everything(2, seed_engines={"random"})  # only reseed Python random
        >>> b_py, b_np = random.random(), np.random.random()
        >>> _ = seed_everything(1)                           # back to fully reset
        >>> c_py, c_np = random.random(), np.random.random()
        >>> a_py == c_py and a_np == c_np     # both engines match under the same full reset
        True
        >>> b_py == a_py                      # Python rng was reseeded to 2, so differs from run a
        False
        >>> b_np == a_np                      # numpy rng was NOT reseeded, but since seed_everything(2)
        ...                                   # only touched random, numpy kept advancing from the state
        ...                                   # seed_everything(1) left it in — not guaranteed equal here
        False
    """

    if seed_engines is None:
        seed_engines = set(_SEED_FUNCTIONS.keys())

    if seed is None:
        if "PL_GLOBAL_SEED" in os.environ:
            seed = int(os.environ["PL_GLOBAL_SEED"])
        else:
            # Draw from an entropy source that is independent of the RNGs this function reseeds —
            # otherwise successive `seed_everything()` calls after a prior seeding would produce
            # the same "fresh" seed.
            seed = secrets.randbits(32)

    for s in seed_engines:
        _SEED_FUNCTIONS[s](seed)

    return seed


class SeedableMixin:
    """This class provides easy utilities to reliably seed stochastic processes.

    This seeding can be used to ensure reproducibility in experiments, both in individual examples with an
    integral seed or in a stochastic process both at a per-event level and at a whole process level by seeding
    with ``None``, in which case a new seed is chosen for each event in the process based on the prior seed
    and stored.

    Examples:
        Calling ``_seed(n)`` twice with the same ``n`` freezes subsequent random draws to the same sequence:

        >>> class M(SeedableMixin):
        ...     def gen(self):
        ...         return (random.random(), np.random.rand())
        >>> m = M()
        >>> _ = m._seed(1); a1, a2 = m.gen(), m.gen()
        >>> _ = m._seed(1); b1, b2 = m.gen(), m.gen()
        >>> a1 == b1 and a2 == b2
        True
        >>> a1 != a2                 # repeated draws within one seeded run still differ
        True

        With ``_seed()`` (no arg), the new seed is drawn from the current random state — so the *sequence*
        of seeds is itself reproducible once the chain has been seeded:

        >>> _ = m._seed(1)
        >>> chain_a = [m._seed() for _ in range(3)]
        >>> _ = m._seed(1)
        >>> chain_b = [m._seed() for _ in range(3)]
        >>> chain_a == chain_b
        True
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
            >>> class M(SeedableMixin):
            ...     @SeedableMixin.WithSeed
            ...     def draw(self):
            ...         return random.random()
            ...     @SeedableMixin.WithSeed(key="custom")
            ...     def keyed_draw(self):
            ...         return random.random()
            >>> m = M()
            >>> m.draw(seed=1) == m.draw(seed=1)
            True
            >>> m.draw(seed=1) != m.draw(seed=2)
            True

        The auto-keyed form records the seed under the function name; the explicit form under ``key=``:
            >>> _ = m.draw(seed=1)
            >>> _ = m.keyed_draw(seed=2)
            >>> _, last_draw = m._last_seed("draw")
            >>> _, last_custom = m._last_seed("custom")
            >>> last_draw, last_custom
            (1, 2)
        """
        if key is None:
            key = fn.__name__

        @functools.wraps(fn)
        def wrapper_seeding(self, *args, seed: int | None = None, **kwargs):
            self._seed(seed=seed, key=key)
            return fn(self, *args, **kwargs)

        return wrapper_seeding
