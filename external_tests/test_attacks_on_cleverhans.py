# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import pytest
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import tensorflow as tf

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.attacks import ElasticNetMethod
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import MadryEtAl
from cleverhans.attacks import FastFeatureAdversaries
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.attacks import LBFGS
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.model import Model as ClModel

from advertorch.attacks import CarliniWagnerL2Attack
from advertorch.attacks import ElasticNetL1Attack
from advertorch.attacks import GradientAttack
from advertorch.attacks import GradientSignAttack
from advertorch.attacks import L2MomentumIterativeAttack
from advertorch.attacks import LinfMomentumIterativeAttack
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import FastFeatureAttack
from advertorch.attacks import LinfBasicIterativeAttack
from advertorch.attacks import L2BasicIterativeAttack
from advertorch.attacks import LBFGSAttack
from advertorch.attacks import JacobianSaliencyMapAttack
from advertorch.test_utils import SimpleModel
from advertorch.test_utils import merge2dicts


BATCH_SIZE = 9
DIM_INPUT = 15
NUM_CLASS = 5
EPS = 0.08

ATOL = 1e-4
RTOL = 1e-4
NB_ITER = 5

# XXX: carlini still doesn't pass sometimes under certain random seed
seed = 66666
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
tf.set_random_seed(seed)
inputs = np.random.uniform(0, 1, size=(BATCH_SIZE, DIM_INPUT))
targets = np.random.randint(0, NUM_CLASS, size=BATCH_SIZE)


targets_onehot = np.zeros((BATCH_SIZE, NUM_CLASS), dtype='int')
targets_onehot[np.arange(BATCH_SIZE), targets] = 1


class SimpleModelTf(ClModel):

    def __init__(self, dim_input, num_classes, session=None):
        import keras
        self.sess = session
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_shape=(dim_input, )))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dense(num_classes))
        self.model = model
        self.flag_weight_set = False

    def set_weights(self, weights):
        self.model.set_weights(weights)
        self.flag_weight_set = True

    def load_state_dict(self, w):
        self.set_weights([
            w['fc1.weight'].cpu().numpy().transpose(),
            w['fc1.bias'].cpu().numpy(),
            w['fc2.weight'].cpu().numpy().transpose(),
            w['fc2.bias'].cpu().numpy(),
        ])

    def get_logits(self, data):
        assert self.flag_weight_set, "Weight Not Set!!!"
        return self.model(data)

    def get_probs(self, data):
        assert self.flag_weight_set, "Weight Not Set!!!"
        return tf.nn.softmax(logits=self.model(data))


def load_weights_pt(model_pt, layers):
    w = model_pt.state_dict()
    layers[0].W = tf.Variable(tf.convert_to_tensor(
        w['fc1.weight'].cpu().numpy().transpose(), tf.float32))
    layers[0].b = tf.Variable(tf.convert_to_tensor(
        w['fc1.bias'].cpu().numpy(), tf.float32))
    layers[2].W = tf.Variable(tf.convert_to_tensor(
        w['fc2.weight'].cpu().numpy().transpose(), tf.float32))
    layers[2].b = tf.Variable(tf.convert_to_tensor(
        w['fc2.bias'].cpu().numpy(), tf.float32))


def setup_simple_model_tf(model_pt, input_shape):
    from cleverhans_tutorials.tutorial_models import MLP, Linear, ReLU
    layers = [Linear(10),
              ReLU(),
              Linear(10)]
    layers[0].name = 'fc1'
    layers[1].name = 'relu'
    layers[2].name = 'fc2'
    model = MLP(layers, input_shape)
    load_weights_pt(model_pt, layers)
    return model



# kwargs for attacks to be tested
attack_kwargs = {
    GradientSignAttack: {
        "cl_class": FastGradientMethod,
        "kwargs": dict(
            eps=EPS,
            clip_min=0.0,
            clip_max=1.0,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            ord=np.inf,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    GradientAttack: {
        "cl_class": FastGradientMethod,
        "kwargs": dict(
            eps=EPS,
            clip_min=0.0,
            clip_max=1.0,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            ord=2,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    LinfPGDAttack: {
        "cl_class": MadryEtAl,
        "kwargs": dict(
            eps=EPS,
            eps_iter=0.01,
            clip_min=0.0,
            clip_max=1.0,
            rand_init=False,
            nb_iter=NB_ITER,
        ),
        "at_kwargs": dict(
            targeted=True,
        ),
        "cl_kwargs": dict(
            ord=np.inf,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    L2MomentumIterativeAttack: {
        "cl_class": MomentumIterativeMethod,
        "kwargs": dict(
            eps=EPS,
            eps_iter=0.01,
            clip_min=0.0,
            clip_max=1.0,
            decay_factor=1.,
            nb_iter=NB_ITER,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            ord=2,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    LinfMomentumIterativeAttack: {
        "cl_class": MomentumIterativeMethod,
        "kwargs": dict(
            eps=EPS,
            eps_iter=0.01,
            clip_min=0.0,
            clip_max=1.0,
            decay_factor=1.,
            nb_iter=NB_ITER,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            ord=np.inf,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    CarliniWagnerL2Attack: {
        "cl_class": CarliniWagnerL2,
        "kwargs": dict(
            max_iterations=100,
            clip_min=0,
            clip_max=1,
            binary_search_steps=9,
            learning_rate=0.1,
            confidence=0.1,
        ),
        "at_kwargs": dict(
            num_classes=NUM_CLASS,
        ),
        "cl_kwargs": dict(
            batch_size=BATCH_SIZE,
            initial_const=1e-3,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    ElasticNetL1Attack: {
        "cl_class": ElasticNetMethod,
        "kwargs": dict(
            max_iterations=100,
            clip_min=0,
            clip_max=1,
            binary_search_steps=9,
            learning_rate=0.1,
            confidence=0.1,
        ),
        "at_kwargs": dict(
            num_classes=NUM_CLASS,
        ),
        "cl_kwargs": dict(
            batch_size=BATCH_SIZE,
            initial_const=1e-3,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    FastFeatureAttack: {
        "cl_class": FastFeatureAdversaries,
        "kwargs": dict(
            nb_iter=NB_ITER,
            clip_min=0,
            clip_max=1,
            eps_iter=0.05,
            eps=0.3,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            layer='logits',
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    LinfBasicIterativeAttack: {
        "cl_class": BasicIterativeMethod,
        "kwargs": dict(
            clip_min=0,
            clip_max=1,
            eps_iter=0.05,
            eps=0.1,
            nb_iter=NB_ITER,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            ord=np.inf,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    L2BasicIterativeAttack: {
        "cl_class": BasicIterativeMethod,
        "kwargs": dict(
            clip_min=0,
            clip_max=1,
            eps_iter=0.05,
            eps=0.1,
            nb_iter=NB_ITER,
        ),
        "at_kwargs": dict(
        ),
        "cl_kwargs": dict(
            ord=2,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    LBFGSAttack: {
        "cl_class": LBFGS,
        "kwargs": dict(
            clip_min=0.,
            clip_max=1.,
            # set binary search step = 3, which can successfully create
            # adversarial images and the difference between advertorch
            # and cleverhans is within the threshold
            # the difference of the two results are very small at first
            # because of some rounding and calculating difference
            # with tensors and numpy arrays, the difference gets larger
            # with more iterations
            binary_search_steps=3,
            max_iterations=50,
            initial_const=1e-3,
            batch_size=BATCH_SIZE,
        ),
        "at_kwargs": dict(
            num_classes=NUM_CLASS,
        ),
        "cl_kwargs": dict(
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
    JacobianSaliencyMapAttack: {
        "cl_class": SaliencyMapMethod,
        "kwargs": dict(
            clip_min=0.0,
            clip_max=1.0,
            theta=1.0,
            gamma=1.0,
        ),
        "at_kwargs": dict(
            num_classes=NUM_CLASS,
            comply_cleverhans=True,
        ),
        "cl_kwargs": dict(
            # nb_classes=NUM_CLASS,
        ),
        "thresholds": dict(
            atol=ATOL,
            rtol=RTOL,
        ),
    },
}


def overwrite_fastfeature(attack, x, g, eta, **kwargs):
    # overwrite cleverhans generate function for fastfeatureattack to
    # allow eta as an input
    from cleverhans.utils_tf import clip_eta

    # Parse and save attack-specific parameters
    assert attack.parse_params(**kwargs)

    g_feat = attack.model.get_layer(g, attack.layer)

    # Initialize loop variables
    eta = tf.Variable(tf.convert_to_tensor(eta, np.float32))
    eta = clip_eta(eta, attack.ord, attack.eps)

    for i in range(attack.nb_iter):
        eta = attack.attack_single_step(x, eta, g_feat)

    # Define adversarial example (and clip if necessary)
    adv_x = x + eta
    if attack.clip_min is not None and attack.clip_max is not None:
        adv_x = tf.clip_by_value(adv_x, attack.clip_min, attack.clip_max)

    return adv_x


def genenerate_ptb_pt(adversary, inputs, targets, delta=None):
    if inputs.ndim == 4:
        # TODO: move the transpose to a better place
        input_t = torch.from_numpy(inputs.transpose(0, 3, 1, 2))
    else:
        input_t = torch.from_numpy(inputs)
    input_t = input_t.float()

    if targets is None:
        adversary.targeted = False
        adv_pt = adversary.perturb(input_t, None)
    else:
        target_t = torch.from_numpy(targets)
        if isinstance(adversary, FastFeatureAttack):
            adv_pt = adversary.perturb(input_t, target_t,
                                       delta=torch.from_numpy(delta))
        else:
            adversary.targeted = True
            adv_pt = adversary.perturb(input_t, target_t)

    adv_pt = adv_pt.cpu().detach().numpy()

    if inputs.ndim == 4:
        # TODO: move the transpose to a better place
        adv_pt = adv_pt.transpose(0, 2, 3, 1)
    return adv_pt - inputs


def compare_at_cl(ptb_at, ptb_cl, atol, rtol):
    assert np.allclose(ptb_at, ptb_cl, atol=atol, rtol=rtol), \
        (np.abs(ptb_at - ptb_cl).max())


def compare_attacks(key, item, targeted=False):
    AdvertorchAttack = key
    CleverhansAttack = item["cl_class"]
    cl_kwargs = merge2dicts(item["kwargs"], item["cl_kwargs"])
    at_kwargs = merge2dicts(item["kwargs"], item["at_kwargs"])
    thresholds = item["thresholds"]
    seed = 6666
    torch.manual_seed(seed)
    np.random.seed(seed)

    # WARNING: don't use tf.InteractiveSession() here
    # It causes that fastfeature attack has to be the last test for some reason
    with tf.Session() as sess:
        model_pt = SimpleModel(DIM_INPUT, NUM_CLASS)
        model_tf = SimpleModelTf(DIM_INPUT, NUM_CLASS)
        model_tf.load_state_dict(model_pt.state_dict())
        adversary = AdvertorchAttack(model_pt, **at_kwargs)

        if AdvertorchAttack is FastFeatureAttack:
            model_tf_fastfeature = setup_simple_model_tf(
                model_pt, inputs.shape)
            delta = np.random.uniform(
                -item["kwargs"]['eps'], item["kwargs"]['eps'],
                size=inputs.shape).astype('float32')
            inputs_guide = np.random.uniform(
                0, 1, size=(BATCH_SIZE, DIM_INPUT)).astype('float32')
            inputs_tf = tf.convert_to_tensor(inputs, np.float32)
            inputs_guide_tf = tf.convert_to_tensor(inputs_guide, np.float32)
            attack = CleverhansAttack(model_tf_fastfeature)
            cl_result = overwrite_fastfeature(attack,
                                              x=inputs_tf,
                                              g=inputs_guide_tf,
                                              eta=delta,
                                              **cl_kwargs)
            init = tf.global_variables_initializer()
            sess.run(init)
            ptb_cl = sess.run(cl_result) - inputs
            ptb_at = genenerate_ptb_pt(
                adversary, inputs, inputs_guide, delta=delta)

        else:
            attack = CleverhansAttack(model_tf, sess=sess)
            if targeted:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ptb_cl = attack.generate_np(
                        inputs, y_target=targets_onehot, **cl_kwargs) - inputs
                ptb_at = genenerate_ptb_pt(adversary, inputs, targets=targets)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ptb_cl = attack.generate_np(
                        inputs, y=None, **cl_kwargs) - inputs
                ptb_at = genenerate_ptb_pt(adversary, inputs, targets=None)

        if AdvertorchAttack is CarliniWagnerL2Attack:
            assert np.sum(np.abs(ptb_at)) > 0 and np.sum(np.abs(ptb_cl)) > 0, \
                ("Both advertorch and cleverhans returns zero perturbation"
                 " of CarliniWagnerL2Attack, "
                 "the test results are not reliable,"
                 " Adjust your testing parameters to avoid this."
                 )
        compare_at_cl(ptb_at, ptb_cl, **thresholds)


@pytest.mark.parametrize("targeted", [False, True])
def test_fgsm_attack(targeted):
    compare_attacks(
        GradientSignAttack,
        attack_kwargs[GradientSignAttack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_fgm_attack(targeted):
    compare_attacks(
        GradientAttack,
        attack_kwargs[GradientAttack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_l2_momentum_iterative_attack(targeted):
    compare_attacks(
        L2MomentumIterativeAttack,
        attack_kwargs[L2MomentumIterativeAttack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_linf_momentum_iterative_attack(targeted):
    compare_attacks(
        LinfMomentumIterativeAttack,
        attack_kwargs[LinfMomentumIterativeAttack],
        targeted)


@pytest.mark.skip(reason="XXX: temporary")
def test_fastfeature_attack():
    compare_attacks(
        FastFeatureAttack,
        attack_kwargs[FastFeatureAttack])


@pytest.mark.parametrize("targeted", [False, True])
def test_pgd_attack(targeted):
    compare_attacks(
        LinfPGDAttack,
        attack_kwargs[LinfPGDAttack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_iterative_sign_attack(targeted):
    compare_attacks(
        LinfBasicIterativeAttack,
        attack_kwargs[LinfBasicIterativeAttack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_iterative_attack(targeted):
    compare_attacks(
        L2BasicIterativeAttack,
        attack_kwargs[L2BasicIterativeAttack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_carlini_l2_attack(targeted):
    compare_attacks(
        CarliniWagnerL2Attack,
        attack_kwargs[CarliniWagnerL2Attack],
        targeted)


@pytest.mark.parametrize("targeted", [False, True])
def test_elasticnet_l1_attack(targeted):
    compare_attacks(
        ElasticNetL1Attack,
        attack_kwargs[ElasticNetL1Attack],
        targeted)


def test_lbfgs_attack():
    compare_attacks(
        LBFGSAttack,
        attack_kwargs[LBFGSAttack],
        True)


@pytest.mark.skip(reason="XXX: temporary")
def test_jsma():
    compare_attacks(
        JacobianSaliencyMapAttack,
        attack_kwargs[JacobianSaliencyMapAttack],
        True)


if __name__ == '__main__':
    # pass
    test_iterative_attack(False)
    test_iterative_attack(True)
