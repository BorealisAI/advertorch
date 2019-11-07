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

# attack type: DDNL2Attack
# attack kwargs: nb_iter=1000
#                gamma=0.05
#                init_norm=1.0
#                quantize=True
#                levels=256
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%
# Among successful attacks (L2 norm) on correctly classified examples:
#    minimum distance: 0.006792
#    median distance: 1.388
#    maximum distance: 3.3
#    average distance: 1.38
#    distance standard deviation: 0.4716

# attack type: DDNL2Attack
# attack kwargs: nb_iter=1000
#                gamma=0.05
#                init_norm=1.0
#                quantize=True
#                levels=256
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 100.0%
# Among successful attacks (L2 norm) on correctly classified examples:
#    minimum distance: 0.005546
#    median distance: 1.872
#    maximum distance: 20.45
#    average distance: 1.917
#    distance standard deviation: 0.733


from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import DDNL2Attack

from advertorch_examples.benchmark_utils import benchmark_margin

batch_size = 100
device = "cuda"

lst_attack = [
    (DDNL2Attack, dict(
        nb_iter=1000, gamma=0.05, init_norm=1., quantize=True, levels=256,
        clip_min=0., clip_max=1., targeted=False)),
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
        lst_benchmark.append(benchmark_margin(
            model, loader, attack_class, attack_kwargs, norm=2, device="cuda"))

print(info)
for item in lst_benchmark:
    print(item)
