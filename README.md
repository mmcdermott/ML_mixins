# ML Mixins

[![PyPI - Version](https://img.shields.io/pypi/v/ml-mixins)](https://pypi.org/project/ml-mixins/)
[![codecov](https://codecov.io/gh/mmcdermott/ML_mixins/graph/badge.svg?token=T2QNDROZ61)](https://codecov.io/gh/mmcdermott/ML_mixins)
[![tests](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yml)
[![code-quality](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/ML_mixins#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/ML_mixins/pulls)
[![contributors](https://img.shields.io/github/contributors/mmcdermott/ML_mixins.svg)](https://github.com/mmcdermott/ML_mixins/graphs/contributors)

## Installation

this package can be installed via [`pip`](https://pypi.org/project/ml-mixins/):

```
pip install ml-mixins
```

Then

```
from mixins import SeedableMixin
...
```

## Description

Useful Python Mixins for ML. These are python mixin classes that can be used to add useful bits of discrete
functionality to python objects for use in ML / data science. They currently include:

1. `SeedableMixin` which adds seeding capabilities, including functions to seed various stages of computation
    in a manner that is both random but also reproducible from a global seed, as well as to store seeds used at
    various times so that a subsection of the computation can be reproduced exactly during debugging outside of
    the rest of the computation flow.
2. `TimeableMixin` adds functionality for timing sections of code for benchmarking performance.

None of these are guaranteed to work or be useful at this point.
