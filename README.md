# ML Mixins

[![PyPI - Version](https://img.shields.io/pypi/v/ml-mixins)](https://pypi.org/project/ml-mixins/)
[![codecov](https://codecov.io/gh/mmcdermott/ML_mixins/graph/badge.svg?token=T2QNDROZ61)](https://codecov.io/gh/mmcdermott/ML_mixins)
[![tests](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yaml)
[![code-quality](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/ML_mixins#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/ML_mixins/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/ML_mixins.svg)](https://github.com/mmcdermott/ML_mixins/graphs/contributors)

This package contains some useful python mixin classes for use in ML / data science, including a mixin for
seeding (`SeedableMixin`), timing (`TimeableMixin`), and memory tracking (`MemTrackableMixin`).

> Every code example below is an executable doctest. `pytest` runs this file directly
> (`--doctest-glob='*.md'`), so if an example here ever drifts from the implementation, CI fails.

## Installation

This package can be installed via [`pip`](https://pypi.org/project/ml-mixins/):

```bash
pip install ml-mixins
```

Memory tracking depends on [`memray`](https://github.com/bloomberg/memray) and is an optional extra:

```bash
pip install 'ml-mixins[memtrackable]'
```

## Usage

You can use these mixins either by (1) defining your classes to inherit from them, then leveraging their
included methods in your derived class, or (2) adding them post-hoc to an existing class for use in secondary
applications such as benchmarking or debugging without overhead in production code.

### `SeedableMixin`

Provides reproducible seeding across `random`, `numpy`, and (if installed) `torch`. The `WithSeed` decorator
exposes an optional `seed` kwarg on any decorated method that reseeds the relevant RNGs before each call,
and every seed used is recorded with a key and timestamp in `self._past_seeds` for later inspection.

```pycon
>>> import random
>>> from mixins import SeedableMixin
>>> class MyModel(SeedableMixin):
...     @SeedableMixin.WithSeed
...     def fit(self, X, y):
...         # "fit" is used as the seed key automatically.
...         return random.random()
...     @SeedableMixin.WithSeed(key="sample")
...     def sample(self, n):
...         # Seeds are recorded under the explicit key "sample".
...         return [random.random() for _ in range(n)]
>>> model = MyModel()
>>> model.fit([0], [0], seed=1) == model.fit([0], [0], seed=1)
True
>>> model.fit([0], [0], seed=1) != model.fit([0], [0], seed=2)
True
>>> _ = model.sample(3)
>>> idx, last = model._last_seed("sample")
>>> isinstance(last, int)
True

```

Seed every registered RNG at once (useful at program startup) via the free function, which returns the seed
it used:

```pycon
>>> from mixins.seedable import seed_everything
>>> seed_everything(0)
0
>>> seed_everything(0, seed_engines={"numpy"})  # restrict to specific engines
0

```

When `torch` is not installed, the `"torch"` engine is simply absent from the default set.

### `TimeableMixin`

Tracks wall-clock durations of methods and arbitrary code blocks, and renders a human-readable summary with
units chosen automatically per key (μs / ms / sec / min / hour / days / weeks).

Three entry points:

- `@TimeableMixin.TimeAs` (with optional `key=...`) — decorator for timed methods.
- `self._time_as(key)` — context manager for timing an inline block.
- `self._register_start(key)` / `self._register_end(key)` — manual control, e.g. across async boundaries.

```pycon
>>> from mixins import TimeableMixin
>>> class MyModel(TimeableMixin):
...     @TimeableMixin.TimeAs                    # auto-keyed as "train"
...     def train(self):
...         pass
...     @TimeableMixin.TimeAs(key="evaluation")
...     def evaluate(self):
...         pass
...     def step(self):
...         with self._time_as("step"):
...             pass
>>> model = MyModel()
>>> for _ in range(3): model.train()
>>> model.evaluate()
>>> for _ in range(5): model.step()
>>> len(model._times_for("train")), len(model._times_for("evaluation")), len(model._times_for("step"))
(3, 1, 5)
>>> all(d >= 0 for d in model._times_for("train"))
True
>>> sorted(model._duration_stats.keys())
['evaluation', 'step', 'train']

```

The summary is ordered by total time spent (mean × count) ascending, so the hotspots appear last. Pass
`only_keys=` to restrict to a subset:

```pycon
>>> out = model._profile_durations()
>>> "train:" in out and "evaluation:" in out and "step:" in out
True
>>> out = model._profile_durations(only_keys={"train"})
>>> out.startswith("train:") and "evaluation" not in out
True

```

### `MemTrackableMixin`

Same shape as `TimeableMixin` but for peak memory usage, powered by
[`memray`](https://github.com/bloomberg/memray). Requires the `memtrackable` extra:

```bash
pip install 'ml-mixins[memtrackable]'
```

Three entry points:

- `@MemTrackableMixin.TrackMemoryAs` (with optional `key=...`) — decorator for tracked methods.
- `self._track_memory_as(key)` — context manager for tracking an inline block.
- `MemTrackableMixin.get_memray_stats(tracker_fp, stats_fp)` — staticmethod to extract stats from an
  existing memray tracker file.

```pycon
>>> import numpy as np
>>> from mixins import MemTrackableMixin
>>> class MyModel(MemTrackableMixin):
...     @MemTrackableMixin.TrackMemoryAs          # auto-keyed as "load"
...     def load(self, n):
...         return np.ones((n,), dtype=np.float64)
...     @MemTrackableMixin.TrackMemoryAs(key="work")
...     def compute(self, n):
...         return np.ones((n, n), dtype=np.float64)
...     def step(self, n):
...         with self._track_memory_as("step"):
...             np.ones((n,), dtype=np.float64)
>>> model = MyModel()
>>> model.load(1_000).shape
(1000,)
>>> model.compute(100).shape
(100, 100)
>>> for _ in range(3): model.step(1_000)
>>> len(model._peak_mem_for("load")), len(model._peak_mem_for("work")), len(model._peak_mem_for("step"))
(1, 1, 3)
>>> all(b > 0 for b in model._peak_mem_for("load"))
True
>>> sorted(model._memory_stats.keys())
['load', 'step', 'work']
>>> out = model._profile_memory_usages()
>>> "load:" in out and "work:" in out and "step:" in out
True

```

Peak memory is read from memray's JSON stats export per tracked call. The summary is ordered by total
memory use (mean × count) ascending, matching `TimeableMixin`.

### Adding Mixins Post-Hoc

When you cannot (or do not want to) change the inheritance of an existing class — for instance, when
instrumenting a third-party model only during benchmarking — `add_mixin` produces a new subclass with the
mixin applied and selected methods decorated, leaving the original class untouched:

```pycon
>>> from mixins import TimeableMixin, add_mixin
>>> class MyModel:
...     def fit(self, X, y):
...         return "fit"
>>> TimedModel = add_mixin(
...     MyModel,
...     TimeableMixin,
...     methods_to_decorate={"fit": TimeableMixin.TimeAs},
... )
>>> TimedModel.__name__                # "Mixin" is stripped, then prefixed
'TimeableMyModel'
>>> model = TimedModel()
>>> model.fit([0], [0])                # behaves like the original
'fit'
>>> len(model._times_for("fit")) == 1  # but is now timed
True

```

Only methods defined directly on the base class are candidates for decoration — inherited methods are
unchanged. `add_mixin` raises `ValueError` if a name in `methods_to_decorate` is missing from the base
class, is not callable, or maps to a non-callable decorator:

```pycon
>>> add_mixin(MyModel, TimeableMixin, {"missing": TimeableMixin.TimeAs})
Traceback (most recent call last):
    ...
ValueError: Method missing not found in class MyModel
>>> add_mixin(MyModel, TimeableMixin, {"fit": "not-a-callable"})
Traceback (most recent call last):
    ...
ValueError: Decorator for fit is not callable!

```

## License

Released under the [MIT License](./LICENSE).
