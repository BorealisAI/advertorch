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
# release: 5.8.0-63-generic
# version: #71~20.04.1-Ubuntu SMP Thu Jul 15 17:46:08 UTC 2021
# machine: x86_64
# python: 3.8.5
# torch: 1.9.0+cu102
# torchvision: 0.10.0+cu102
# advertorch: 0.2.4
#
# attack type: DeepfoolLinfAttack
# attack kwargs: eps=0.3
#                nb_iter=50
#                overshoot=0.02
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%
# Among successful attacks (2 norm) on correctly classified examples:
#    minimum distance: 0.004306
#    median distance: 2.356
#    maximum distance: 5.376
#    average distance: 2.33
#    distance standard deviation: 0.8195
#
# attack type: DeepfoolLinfAttack
# attack kwargs: eps=0.3
#                nb_iter=150
#                overshoot=0.02
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%
# Among successful attacks (2 norm) on correctly classified examples:
#    minimum distance: 0.004306
#    median distance: 2.356
#    maximum distance: 5.376
#    average distance: 2.33
#    distance standard deviation: 0.8195
#
# attack type: DeepfoolLinfAttack
# attack kwargs: eps=0.3
#                nb_iter=50
#                overshoot=0.02
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 7.42%
# Among successful attacks (2 norm) on correctly classified examples:
#    minimum distance: 0.02558
#    median distance: 3.37
#    maximum distance: 6.413
#    average distance: 3.199
#    distance standard deviation: 1.395
#
# attack type: DeepfoolLinfAttack
# attack kwargs: eps=0.3
#                nb_iter=150
#                overshoot=0.02
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 8.41%
# Among successful attacks (2 norm) on correctly classified examples:
#    minimum distance: 0.02558
#    median distance: 3.804
#    maximum distance: 6.413
#    average distance: 3.422
#    distance standard deviation: 1.416


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
        eps=0.3, nb_iter=50, overshoot=0.02, clip_min=0., clip_max=1.)),
    (DeepfoolLinfAttack, dict(
        eps=0.3, nb_iter=150, overshoot=0.02, clip_min=0., clip_max=1.)),
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
