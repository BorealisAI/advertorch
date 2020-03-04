# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
from pathlib import Path

import tensorflow as tf
import torch

from advertorch.bpda import BPDAWrapper
from advertorch_examples.utils import ROOT_PATH, mkdir

MODEL_PATH = os.path.join(ROOT_PATH, "madry_et_al_models")
mkdir(MODEL_PATH)
Path(os.path.join(MODEL_PATH, "__init__.py")).touch()
sys.path.append(MODEL_PATH)


class WrappedTfModel(object):

    def __init__(self, weights_path, model_class):

        model = model_class()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config).__enter__()
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(weights_path)
        saver.restore(sess, checkpoint)

        self.inputs = model.x_input
        self.logits = model.pre_softmax

        self.session = tf.get_default_session()
        assert self.session.graph == self.inputs.graph

        with self.session.graph.as_default():
            self.bw_gradient_pre = tf.placeholder(
                tf.float32, self.logits.shape)
            bw_loss = tf.reduce_sum(self.logits * self.bw_gradient_pre)
            self.bw_gradients = tf.gradients(bw_loss, self.inputs)[0]

    def backward(self, inputs_val, logits_grad_val):
        inputs_grad_val = self.session.run(
            self.bw_gradients,
            feed_dict={
                self.inputs: inputs_val,
                self.bw_gradient_pre: logits_grad_val,
            })
        return inputs_grad_val

    def forward(self, inputs_val):
        logits_val = self.session.run(
            self.logits,
            feed_dict={
                self.inputs: inputs_val,
            })
        return logits_val


class TorchWrappedModel(object):

    def __init__(self, tfmodel, device):
        self.tfmodel = tfmodel
        self.device = device

    def _to_numpy(self, val):
        return val.cpu().detach().numpy()

    def _to_torch(self, val):
        return torch.from_numpy(val).float().to(self.device)

    def forward(self, inputs_val):
        rval = self.tfmodel.forward(self._to_numpy(inputs_val))
        return self._to_torch(rval)

    def backward(self, inputs_val, logits_grad_val):
        rval = self.tfmodel.backward(
            self._to_numpy(inputs_val),
            self._to_numpy(logits_grad_val),
        )
        return self._to_torch(rval)



def get_madry_et_al_tf_model(dataname, device="cuda"):
    if dataname == "MNIST":
        weights_path = os.path.join(
            MODEL_PATH, 'mnist_challenge/models/secret')

        try:
            from mnist_challenge.model import Model
            print("mnist_challenge found and imported")
        except (ImportError, ModuleNotFoundError):
            print("mnist_challenge not found, downloading ...")
            os.system("bash download_mnist_challenge.sh {}".format(MODEL_PATH))
            from mnist_challenge.model import Model
            print("mnist_challenge found and imported")

        def _process_inputs_val(val):
            return val.view(val.shape[0], 784)

        def _process_grads_val(val):
            return val.view(val.shape[0], 1, 28, 28)


    elif dataname == "CIFAR10":
        weights_path = os.path.join(
            MODEL_PATH, 'cifar10_challenge/models/model_0')

        try:
            from cifar10_challenge.model import Model
            print("cifar10_challenge found and imported")
        except (ImportError, ModuleNotFoundError):
            print("cifar10_challenge not found, downloading ...")
            os.system(
                "bash download_cifar10_challenge.sh {}".format(MODEL_PATH))
            from cifar10_challenge.model import Model
            print("cifar10_challenge found and imported")

        from functools import partial
        Model = partial(Model, mode="eval")

        def _process_inputs_val(val):
            return 255. * val.permute(0, 2, 3, 1)

        def _process_grads_val(val):
            return val.permute(0, 3, 1, 2) / 255.

    else:
        raise ValueError(dataname)


    def _wrap_forward(forward):
        def new_forward(inputs_val):
            return forward(_process_inputs_val(inputs_val))
        return new_forward

    def _wrap_backward(backward):
        def new_backward(inputs_val, logits_grad_val):
            return _process_grads_val(backward(
                _process_inputs_val(*inputs_val), *logits_grad_val))
        return new_backward


    ptmodel = TorchWrappedModel(
        WrappedTfModel(weights_path, Model), device)
    model = BPDAWrapper(
        forward=_wrap_forward(ptmodel.forward),
        backward=_wrap_backward(ptmodel.backward)
    )

    return model
