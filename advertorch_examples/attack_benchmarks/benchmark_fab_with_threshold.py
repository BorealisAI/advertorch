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
# release: 4.4.0-146-generic
# version: #172-Ubuntu SMP Wed Apr 3 09:00:08 UTC 2019
# machine: x86_64
# python: 3.6.8
# torch: 1.4.0
# torchvision: 0.5.0
# advertorch: 0.2.2

# attack type: FABWithThreshold
# attack kwargs: norm=Linf
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                eps=0.4
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 100.0%
# Among successful attacks (Linf norm) on correctly classified examples:
#    minimum distance: 0.0001393
#    median distance: 0.112
#    maximum distance: 0.2132
#    average distance: 0.1092
#    distance standard deviation: 0.03501

# attack type: FABWithThreshold
# attack kwargs: norm=L2
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                eps=2.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 89.1%
# Among successful attacks (L2 norm) on correctly classified examples:
#    minimum distance: 0.001727
#    median distance: 1.359
#    maximum distance: 2.0
#    average distance: 1.309
#    distance standard deviation: 0.4015

# attack type: FABWithThreshold
# attack kwargs: norm=L1
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                eps=10.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.89%
# attack success rate: 70.74%
# Among successful attacks (L1 norm) on correctly classified examples:
#    minimum distance: 0.007687
#    median distance: 6.239
#    maximum distance: 9.997
#    average distance: 6.096
#    distance standard deviation: 2.302

# attack type: FABWithThreshold
# attack kwargs: norm=Linf
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                eps=0.4
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 92.5%
# Among successful attacks (Linf norm) on correctly classified examples:
#    minimum distance: 0.001414
#    median distance: 0.3482
#    maximum distance: 0.3999
#    average distance: 0.3411
#    distance standard deviation: 0.04846

# attack type: FABWithThreshold
# attack kwargs: norm=L2
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                eps=2.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 9.23%
# Among successful attacks (L2 norm) on correctly classified examples:
#    minimum distance: 0.003937
#    median distance: 1.07
#    maximum distance: 1.999
#    average distance: 1.101
#    distance standard deviation: 0.6595

# attack type: FABWithThreshold
# attack kwargs: norm=L1
#                n_restarts=1
#                n_iter=20
#                alpha_max=0.1
#                eta=1.05
#                beta=0.9
#                eps=10.0
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 5.04%
# Among successful attacks (L1 norm) on correctly classified examples:
#    minimum distance: 0.006217
#    median distance: 1.553
#    maximum distance: 9.708
#    average distance: 2.462
#    distance standard deviation: 2.407



from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import FABWithThreshold, FABTargeted

from advertorch_examples.benchmark_utils import benchmark_margin

batch_size = 1000
device = "cuda"

lst_attack = [
    (FABWithThreshold, dict(
        norm='Linf',
        n_restarts=1,
        n_iter=20,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        eps=.4,
        )),
    (FABWithThreshold, dict(
        norm='L2',
        n_restarts=1,
        n_iter=20,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        eps=2.,
        )),
    (FABWithThreshold, dict(
        norm='L1',
        n_restarts=1,
        n_iter=20,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        eps=10.,
        )),
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
