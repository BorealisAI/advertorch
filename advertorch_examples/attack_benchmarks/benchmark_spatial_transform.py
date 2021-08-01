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
# release: 4.15.0-111-generic
# version: #112-Ubuntu SMP Thu Jul 9 20:32:34 UTC 2020
# machine: x86_64
# python: 3.7.3
# torch: 1.6.0.dev20200402+cu101
# torchvision: 0.6.0.dev20200402+cu101
# advertorch: 0.2.3

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 0.0, 'trans': 0.0}
#                random_tries=1
#                attack_type=random
#                num_rot=1
#                num_trans=1
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 1.16%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=10
#                attack_type=grid
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 99.98%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=10
#                attack_type=random
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 75.4%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=50
#                attack_type=random
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 95.93%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 0.0, 'trans': 0.0}
#                random_tries=1
#                attack_type=random
#                num_rot=1
#                num_trans=1
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 1.4%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=10
#                attack_type=grid
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 99.99%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=10
#                attack_type=random
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 70.62%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=50
#                attack_type=random
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 94.12%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 0.0, 'trans': 0.0}
#                random_tries=1
#                attack_type=random
#                num_rot=1
#                num_trans=1
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: cifar10_test, 10000 samples
# model: CIFAR10 ResNet18 standard training
# accuracy: 94.64%
# attack success rate: 7.05%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=10
#                attack_type=grid
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: cifar10_test, 10000 samples
# model: CIFAR10 ResNet18 standard training
# accuracy: 94.64%
# attack success rate: 86.71%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=10
#                attack_type=random
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: cifar10_test, 10000 samples
# model: CIFAR10 ResNet18 standard training
# accuracy: 94.64%
# attack success rate: 57.53%

# attack type: SpatialTransformAttack2
# attack kwargs: loss_fn=CrossEntropyLoss()
#                spatial_constraint={'rot': 30, 'trans': 0.10714285714285714}
#                random_tries=50
#                attack_type=random
#                num_rot=31
#                num_trans=5
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: cifar10_test, 10000 samples
# model: CIFAR10 ResNet18 standard training
# accuracy: 94.64%
# attack success rate: 72.53%

import torch
import torch.nn as nn

from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.utils import get_cifar10_test_loader
from advertorch_examples.utils import get_cifar10_resnet18_clntrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import SpatialTransformAttack2

from advertorch_examples.benchmark_utils import benchmark_attack_success_rate

batch_size = 100
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

lst_attack = [
    (SpatialTransformAttack2,
     dict(loss_fn=nn.CrossEntropyLoss(reduction="sum"),
          spatial_constraint={'rot': 0.0, 'trans': 0.0},
          random_tries=1, attack_type='random',
          num_rot=1, num_trans=1,
          clip_min=0.0, clip_max=1.0, targeted=False)),
    (SpatialTransformAttack2,
     dict(loss_fn=nn.CrossEntropyLoss(reduction="sum"),
          spatial_constraint={'rot': 30, 'trans': 3 / 28},
          random_tries=10, attack_type='grid',
          num_rot=31, num_trans=5,
          clip_min=0.0, clip_max=1.0, targeted=False)),
    (SpatialTransformAttack2,
     dict(loss_fn=nn.CrossEntropyLoss(reduction="sum"),
          spatial_constraint={'rot': 30, 'trans': 3 / 28},
          random_tries=10, attack_type='random',
          num_rot=31, num_trans=5,
          clip_min=0.0, clip_max=1.0, targeted=False)),
    (SpatialTransformAttack2,
     dict(loss_fn=nn.CrossEntropyLoss(reduction="sum"),
          spatial_constraint={'rot': 30, 'trans': 3 / 28},
          random_tries=50, attack_type='random',
          num_rot=31, num_trans=5,
          clip_min=0.0, clip_max=1.0, targeted=False)),
]  # each element in the list is the tuple (attack_class, attack_kwargs)

mnist_clntrained_model = get_mnist_lenet5_clntrained(device).to(device)
mnist_advtrained_model = get_mnist_lenet5_advtrained(device).to(device)
mnist_test_loader = get_mnist_test_loader(batch_size=batch_size)

cifar10_clntrained_model = get_cifar10_resnet18_clntrained(device)
cifar10_test_loader = get_cifar10_test_loader(batch_size=batch_size)

lst_setting = [
    (mnist_clntrained_model, mnist_test_loader),
    (mnist_advtrained_model, mnist_test_loader),
    # (cifar10_clntrained_model, cifar10_test_loader),  # need trained model
]

info = get_benchmark_sys_info()

lst_benchmark = []
for model, loader in lst_setting:
    for attack_class, attack_kwargs in lst_attack:
        lst_benchmark.append(benchmark_attack_success_rate(
            model, loader, attack_class, attack_kwargs, device=device))

print(info)
for item in lst_benchmark:
    print(item)
