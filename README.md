# ML Mixins

[![PyPI - Version](https://img.shields.io/pypi/v/MEDS-transforms)](https://pypi.org/project/ml-mixins/)
[![codecov](https://codecov.io/gh/mmcdermott/ML_mixins/graph/badge.svg?token=T2QNDROZ61)](https://codecov.io/gh/mmcdermott/ML_mixins)
[![tests](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/tests.yml)
[![code-quality](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/ML_mixins/actions/workflows/code-quality-main.yaml)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
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

1. `SeedableMixin` which adds nice seeding capabilities, including functions to seed various stages of
    computation in a manner that is both random but also reproducible from a global seed, as well as to store
    seeds used at various times so that a subsection of the computation can be reproduced exactly during
    debugging outside of the rest of the computation flow.
2. `TimeableMixin` adds functionality for timing sections of code.
3. `SaveableMixin` adds customizable save/load functionality (using pickle)
4. `SwapcacheableMixin`. This one is a bit more niche. It adds a "_swapcache_" to the class, which allows
    one to store various iterations of parameters keyed by an arbitrary python object with an equality
    operator, with a notion of a "current" setting whose values are then exposed as main class attributes.
    The intended use-case is for data processing classes, where it may be desirable to try different
    preprocesisng settings, have the object retain derived data for those settings, but present a
    front-facing interface that looks like it is only computing a single setting. For example, if running
    tfidf under different stopwords and ngram settings, one can run the system via the swapcache under
    settings A, and the class can present an interface of `[obj].stop_words`, `obj.ngram_range`,
    `obj.tfidf_vectorized_data`, but then this can be transparently updated to a different setting without
    discarding that data via the swapcache interface.
5. `TQDMableMixin`. This one adds a `_tqdm` method to a class which automatically progressbar-ifies ranges
    for iteration, unless the range is sufficiently short or the class has `self.tqdm` set to `None`

None of these are guaranteed to work or be useful at this point.
