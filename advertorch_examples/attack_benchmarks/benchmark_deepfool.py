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
# release: 3.10.0-1062.el7.x86_64
# version: #1 SMP Wed Aug 7 18:08:02 UTC 2019
# machine: x86_64
# python: 3.7.3
# torch: 1.4.0
# torchvision: 0.5.0
# advertorch: 0.2.2

# attack type: DeepfoolLinfAttack
# attack kwargs: nb_iter=50
#                overshoot=0.02
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 32.76%
# Among successful attacks (2 norm) on correctly classified examples:
#    minimum distance: 0.004306
#    median distance: 1.528
#    maximum distance: 2.061
#    average distance: 1.403
#    distance standard deviation: 0.4457

# attack type: DeepfoolLinfAttack
# attack kwargs: nb_iter=50
#                overshoot=0.02
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 2.48%
# Among successful attacks (2 norm) on correctly classified examples:
#    minimum distance: 0.02558
#    median distance: 1.058
#    maximum distance: 2.139
#    average distance: 0.9923
#    distance standard deviation: 0.5285


from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import DeepfoolLinfAttack

from advertorch_examples.benchmark_utils import benchmark_margin

batch_size = 100
device = "cuda"

print('Begin testing...')
lst_attack = [
    (DeepfoolLinfAttack, dict(
        nb_iter=50, overshoot=0.02, clip_min=0., clip_max=1.)),
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
