# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#
#
# Automatically generated benchmark report (screen print of running this file)
#
# sysname: Linux
# release: 4.4.0-140-generic
# version: #166-Ubuntu SMP Wed Nov 14 20:09:47 UTC 2018
# machine: x86_64
# python: 3.7.3
# torch: 1.1.0
# torchvision: 0.3.0
# advertorch: 0.1.5

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=CrossEntropyLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=CrossEntropyLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 6.8%


import torch.nn as nn

from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import LinfPGDAttack
# TODO: from advertorch.attacks import L2BasicIterativeAttack
# TODO: from advertorch.attacks import LinfBasicIterativeAttack
# TODO: from advertorch.attacks import PGDAttack
# TODO: from advertorch.attacks import L2PGDAttack
# TODO: from advertorch.attacks import L1PGDAttack
# TODO: from advertorch.attacks import SparseL1DescentAttack
# TODO: from advertorch.attacks import MomentumIterativeAttack
# TODO: from advertorch.attacks import L2MomentumIterativeAttack
# TODO: from advertorch.attacks import LinfMomentumIterativeAttack
# TODO: from advertorch.attacks import FastFeatureAttack

from advertorch_examples.benchmark_utils import benchmark_attack_success_rate

batch_size = 100
device = "cuda"

lst_attack = [
    (LinfPGDAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False)),
]  # each element in the list is the tuple (attack_class, attack_kwargs)

mnist_clntrained_model = get_mnist_lenet5_clntrained().to(device)
mnist_advtrained_model = get_mnist_lenet5_advtrained().to(device)
mnist_test_loader = get_mnist_test_loader(batch_size=batch_size)

lst_setting = [
    (mnist_clntrained_model, mnist_test_loader),
    (mnist_advtrained_model, mnist_test_loader),
]


info = get_benchmark_sys_info()

lst_benchmark = []
for model, loader in lst_setting:
    for attack_class, attack_kwargs in lst_attack:
        lst_benchmark.append(benchmark_attack_success_rate(
            model, loader, attack_class, attack_kwargs, device="cuda"))

print(info)
for item in lst_benchmark:
    print(item)
