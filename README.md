# AdverTorch - A Toolbox for Adversarial Robustness Research based on PyTorch
## Latest release (v0.1.0)

AdverTorch is a Python toolbox for adversarial robustness research. The primary functionalities are implemented in PyTorch.


## Installation

### Installing AdverTorch itself

We developed AdverTorch under Python 3.6 and PyTorch 1.0.0 & 0.4.1. To install AdverTorch, simply run

```
pip install advertorch
```

or clone the repo and run
```
python setup.py install
```

To install the package in "editable" mode:
```
pip install -e .
```

### Setting up the testing environments

Some attacks are tested against implementations in [Foolbox](https://github.com/bethgelab/foolbox) or [CleverHans](https://github.com/tensorflow/cleverhans) to ensure correctness. Currently, they are tested under the following versions of related libraries.
```
conda install -c anaconda tensorflow-gpu==1.11.0
pip install git+https://github.com/tensorflow/cleverhans.git@336b9f4ed95dccc7f0d12d338c2038c53786ab70
pip install Keras==2.2.2
pip install foolbox==1.3.2
```


## Examples
```python
# prepare your pytorch model as "model"
# prepare a batch of data and label as "cln_data" and "true_label"
# ...

from advertorch.attacks import LinfPGDAttack

adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

adv_untargeted = adversary.perturb(cln_data, true_label)

target = torch.ones_like(true_label) * 3
adversary.targeted = True
adv_targeted = adversary.perturb(cln_data, target)
```

For runnable examples see [`advertorch_examples/tutorial_attack_defense_bpda_mnist.ipynb`](https://github.com/BorealisAI/advertorch/blob/master/advertorch_examples/tutorial_attack_defense_bpda_mnist.ipynb) for how to attack and defend; see [`advertorch_examples/tutorial_train_mnist.py`](https://github.com/BorealisAI/advertorch/blob/master/advertorch_examples/tutorial_train_mnist.py) for how to adversarially train a robust model on MNIST.


## Coming Soon

AdverTorch is still under active development. We will add the following features/items down the road:

* a technical report
* a blog post
* more examples
* more complete documentation in the code
* documentation at https://advertorch.readthedocs.io/en/latest/
* support for other machine learning frameworks, e.g. TensorFlow
* more attacks, defenses and other related functionalities
* support for other Python versions and future PyTorch versions
* contributing guidelines
* ...


## Known issues

`FastFeatureAttack` and `JacobianSaliencyMapAttack` do not pass the tests against the version of CleverHans used. (They use to pass tests on a previous version of CleverHans.) This issue is being investigated. In the file `test_attacks_on_cleverhans.py`, they are marked as "skipped" in `pytest` tests. 

## License

This project is licensed under the LGPL. The terms and conditions can be found in the LICENSE and LICENSE.GPL files.


## Contributors

* [Gavin Weiguang Ding](https://gwding.github.io/)
* Luyu Wang
* Xiaomeng Jin