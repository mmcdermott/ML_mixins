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

You can use these mixins in your own classes by inheriting from them, then leveraging their included methods
in your derived class.

### `SeedableMixin`

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

### `TimeableMixin`

TODO

### `MemTrackableMixin`

TODO
