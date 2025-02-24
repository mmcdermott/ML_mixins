# ML Mixins

[![PyPI - Version](https://img.shields.io/pypi/v/ml-mixins)](https://pypi.org/project/ml-mixins/)
[![codecov](https://codecov.io/gh/mmcdermott/ML_mixins/graph/badge.svg?token=T2QNDROZ61)](https://codecov.io/gh/mmcdermott/ML_mixins)
[![tests](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yml)
[![code-quality](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/ML_mixins#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/ML_mixins/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/ML_mixins.svg)](https://github.com/mmcdermott/ML_mixins/graphs/contributors)

This package contains some useful python mixin classes for use in ML / data science, including a mixin for
seeding (`SeedableMixin`), timing (`TimeableMixin`), and memory tracking (`MemTrackableMixin`).

## Installation

this package can be installed via [`pip`](https://pypi.org/project/ml-mixins/):

```
pip install ml-mixins
```

## Usage

You can use these mixins either by (1) Defining your classes to inherit from them, then leveraging their
included methods in your derived class, or (2) Adding them post-hoc to an existing class for use in secondary
applications such as benchmarking or debugging without overhead in production code. Below, we show how to use
each mixin directly first, then we show how to add them to an existing class at the end, as that process will
still leverage the same decorator methods and class member variables in the resulting modified classes.

### Mixin Documentation

#### `SeedableMixin`

```python
from mixins import SeedableMixin

class MyModel(SeedableMixin):
    ...

    @SeedableMixin.WithSeed
    def fit(self, X, y):
        # This function can now be called with a seed kwarg, or it will use a pseudo-random seed which will be
        # saved to a class member variable if a seed is not passed.
...
```

#### `TimeableMixin`

TODO

#### `MemTrackableMixin`

TODO

### Adding Mixins Post-Hoc

```python
from mixins import TimeableMixin, add_mixin


class MyModel:
    ...

    def fit(self, X, y): ...


# Add the mixin to the class
TimedModel = add_mixin(
    MyModel, TimeableMixin, decorate_methods={"fit": TimeableMixin.TimeAs}
)

# Now, the class `TimedModel` will have the same methods as `MyModel`, but with the added timing
# functionality:

model = TimedModel()
model.fit(X, y)
model._profile_durations()  # will print durations...
```
