# Contributing to AdverTorch

Thank you considering contributing to AdverTorch!

This document provide brief guidelines for potential contributors.

Please use pull requests for new features, bug fixes, new examples and etc. If you work on something with significant efforts, please mention it in early stage using issues.

We ask that you follow the `PEP8` coding style in your pull requests, [`flake8`](http://flake8.pycqa.org/) is used in continuous integration to enforce this.

---
### Detailed guidelines for contributing new attacks
- *(mandatory)* The implementation file should be added to the folder `advertorch/attacks`, and the class should be imported in `advertorch/attacks/__init__.py`.
- *(mandatory)* The attack should be included in different unit tests, this can be done by adding the attack class to different lists in `advertorch/test_utils.py`
    + add to `general_input_attacks` if it can perturb input tensor of any shape (not limited to images),
    + add to `image_only_attacks` if it only works on images,
    + add to `label_attacks` if the attack manipulates labels,
    + add to `feature_attacks` if the attack manipulates features,
    + add to `batch_consistent_attacks` if the attack's behavior should be the same when it is applied to a single example or a batch,
    + add to `targeted_only_attacks` if the attack is a label attack and does not work for the untargeted case,
    + add entry to `attack_kwargs` in `advertorch/tests/test_attacks_running.py`, for setting the hyperparameters used for test.
- *(mandatory)* Benchmark the attack with at least one performance measure, by adding a script to `advertorch_examples/attack_benchmarks`.
- *(mandatory)* If the contributor has a GPU computer, run `pytest` locally to make sure all the tests pass. (This is because travis-ci currently do not provide GPU machines for continuous integration.) If the contributor does not have a GPU computer, please let us know in the pull request. 
- *(optional)* When an attack can be compared against other implementations, a comparison test could be added to `advertorch/external_tests`.
- *(optional)* Add an ipython notebook example.

---
### Copyright notice at the beginning of files

For files purely contributed by contributors outside of RBC, the following lines should appear at the beginning
```
# Copyright (c) 2019-present, Name of the Contributor.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
```

For files involving both RBC employees and 
contributors outside of RBC, the following lines should appear at the beginning
```
# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
```
