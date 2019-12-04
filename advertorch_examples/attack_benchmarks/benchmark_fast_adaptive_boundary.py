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

# attack type: FABAttack
# attack kwargs: norm=Linf
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                loss_fn=None
# data: mnist_test
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%
# Among successful attacks (Linf norm) on correctly classified examples:
#    minimum distance: 0.0001396
#    median distance: 0.112
#    maximum distance: 0.2155
#    average distance: 0.1092
#    distance standard deviation: 0.03498

# attack type: FABAttack
# attack kwargs: norm=L2
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                loss_fn=None
# data: mnist_test
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%
# Among successful attacks (L2 norm) on correctly classified examples:
#    minimum distance: 0.001726
#    median distance: 1.423
#    maximum distance: 3.01
#    average distance: 1.412
#    distance standard deviation: 0.4805

# attack type: FABAttack
# attack kwargs: norm=L1
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                loss_fn=None
# data: mnist_test
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 99.55%
# Among successful attacks (L1 norm) on correctly classified examples:
#    minimum distance: 0.007688
#    median distance: 7.61
#    maximum distance: 36.06
#    average distance: 8.365
#    distance standard deviation: 4.42

# attack type: FABAttack
# attack kwargs: norm=Linf
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                loss_fn=None
# data: mnist_test
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 99.86%
# Among successful attacks (Linf norm) on correctly classified examples:
#    minimum distance: 0.001405
#    median distance: 0.3509
#    maximum distance: 0.6404
#    average distance: 0.3476
#    distance standard deviation: 0.05255

# attack type: FABAttack
# attack kwargs: norm=L2
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                loss_fn=None
# data: mnist_test
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 98.35%
# Among successful attacks (L2 norm) on correctly classified examples:
#    minimum distance: 0.003942
#    median distance: 3.04
#    maximum distance: 19.92
#    average distance: 3.205
#    distance standard deviation: 1.311

# attack type: FABAttack
# attack kwargs: norm=L1
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                loss_fn=None
# data: mnist_test
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 94.33%
# Among successful attacks (L1 norm) on correctly classified examples:
#    minimum distance: 0.00622
#    median distance: 112.8
#    maximum distance: 441.9
#    average distance: 114.6
#    distance standard deviation: 52.85



from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import FABAttack

from advertorch_examples.benchmark_utils import benchmark_margin

batch_size = 100
device = "cuda"

lst_attack = [
    (FABAttack, dict(
        norm='Linf',
        n_restarts=1,
        n_iter=20,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        loss_fn=None)),
    (FABAttack, dict(
        norm='L2',
        n_restarts=1,
        n_iter=20,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        loss_fn=None)),
    (FABAttack, dict(
        norm='L1',
        n_restarts=1,
        n_iter=20,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        loss_fn=None)),
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
            model, loader, attack_class, attack_kwargs,
            norm=attack_kwargs["norm"]))

print(info)
for item in lst_benchmark:
    print(item)
