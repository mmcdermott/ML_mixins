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
applications such as benchmarking or debugging without overhead in production code. Below, we show how to use
each mixin directly first, then we show how to add them to an existing class at the end.

### Mixin Documentation

#### `SeedableMixin`

Provides reproducible seeding across `random`, `numpy`, and (if installed) `torch`. The `WithSeed` decorator
exposes an optional `seed` kwarg on any decorated method that reseeds the relevant RNGs before each call,
and every seed used is recorded with a key and timestamp in `self._past_seeds` for later inspection.

```python
import random
from mixins import SeedableMixin


class MyModel(SeedableMixin):
    @SeedableMixin.WithSeed
    def fit(self, X, y):
        # Automatically reseeded; the function's own name ("fit") is used as the seed key.
        return random.random()

    @SeedableMixin.WithSeed(key="sample")
    def sample(self, n):
        # Seeds are recorded under the custom key "sample" instead of the method name.
        return [random.random() for _ in range(n)]


model = MyModel()
model.fit([0], [0], seed=1)  # deterministic given seed=1
model.fit([0], [0])  # uses a fresh seed, drawn and recorded
idx, last = model._last_seed("fit")  # retrieve the most recent "fit" seed
```

Seed every registered RNG at once (useful at program startup) via the free function:

```python
from mixins.seedable import seed_everything

seed_everything(0)  # seeds random, numpy, and torch (if installed)
seed_everything(0, seed_engines={"numpy"})  # restrict to specific engines
```

When `torch` is not installed, the `"torch"` engine is simply absent from the default set.

#### `TimeableMixin`

Tracks wall-clock durations of methods and arbitrary code blocks, and renders a human-readable summary with
units chosen automatically per key (μs / ms / sec / min / hour / days / weeks).

Three entry points:

- `@TimeableMixin.TimeAs` (with optional `key=...`) — decorator for timed methods.
- `self._time_as(key)` — context manager for timing an inline block.
- `self._register_start(key)` / `self._register_end(key)` — manual control, e.g. across async boundaries.

```python
import time
from mixins import TimeableMixin


class MyModel(TimeableMixin):
    @TimeableMixin.TimeAs  # auto-keyed as "train"
    def train(self):
        time.sleep(0.05)

    @TimeableMixin.TimeAs(key="evaluation")
    def evaluate(self):
        time.sleep(0.10)

    def step(self):
        with self._time_as("step"):
            time.sleep(0.01)


model = MyModel()
for _ in range(3):
    model.train()
model.evaluate()
for _ in range(5):
    model.step()

print(model._profile_durations())
# evaluation: 100.2 ms
# train:       50.1 ± 0.1 ms (x3)
# step:        10.1 ± 0.1 ms (x5)

print(model._profile_durations(only_keys={"train"}))
# train: 50.1 ± 0.1 ms (x3)

# Raw per-call deltas and in-flight duration are available too:
model._times_for("train")  # list of three floats, seconds
# model._time_so_far("in_progress_key") # wall-clock since the last register_start, for still-open timers
```

The summary is ordered by total time spent (mean × count) ascending, so the hotspots appear last.

#### `MemTrackableMixin`

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

```python
import numpy as np
from mixins import MemTrackableMixin


class MyModel(MemTrackableMixin):
    @MemTrackableMixin.TrackMemoryAs  # auto-keyed as "load"
    def load(self, n):
        return np.ones((n,), dtype=np.float64)

    @MemTrackableMixin.TrackMemoryAs(key="work")
    def compute(self, n):
        return np.ones((n, n), dtype=np.float64)

    def step(self, n):
        with self._track_memory_as("step"):
            np.ones((n,), dtype=np.float64)


model = MyModel()
model.load(1_000_000)
model.compute(1_000)
for _ in range(3):
    model.step(100_000)

print(model._profile_memory_usages())
# step: ...kB (x3)
# load: ...MB
# work: ...MB

# Raw per-call peaks (in bytes):
model._peak_mem_for("load")
```

Peak memory is read from memray's JSON stats export per tracked call. The summary is ordered by total
memory use (mean × count) ascending, matching `TimeableMixin`.

### Adding Mixins Post-Hoc

When you cannot (or do not want to) change the inheritance of an existing class — for instance, when
instrumenting a third-party model only during benchmarking — `add_mixin` produces a new subclass with the
mixin applied and selected methods decorated, leaving the original class untouched:

```python
from mixins import TimeableMixin, add_mixin


class MyModel:
    def fit(self, X, y): ...


# Produce a new class with the mixin applied and `fit` wrapped with @TimeAs.
TimedModel = add_mixin(
    MyModel,
    TimeableMixin,
    methods_to_decorate={"fit": TimeableMixin.TimeAs},
)

model = TimedModel()
model.fit(X, y)
print(model._profile_durations())  # timing data for "fit"
```

The resulting class is named `<MixinPrefix><OriginalName>` (here `TimeableMyModel`). Only methods defined
directly on the base class are candidates for decoration — inherited methods are unchanged. `add_mixin`
raises `ValueError` if a name in `methods_to_decorate` is missing from the base class, is not callable, or
maps to a non-callable decorator.

## License

Released under the [MIT License](./LICENSE).
