# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
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

# attack type: LinfSPSAAttack
# attack kwargs: eps=0.3
#                delta=0.01
#                lr=0.01
#                nb_iter=1000
#                nb_sample=128
#                max_batch_size=64
#                targeted=False
#                loss_fn=None
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 100 samples
# model: MNIST LeNet5 standard training
# accuracy: 99.0%
# attack success rate: 100.0%

# attack type: LinfSPSAAttack
# attack kwargs: eps=0.3
#                delta=0.01
#                lr=0.01
#                nb_iter=100
#                nb_sample=8192
#                max_batch_size=64
#                targeted=False
#                loss_fn=None
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 100 samples
# model: MNIST LeNet5 standard training
# accuracy: 99.0%
# attack success rate: 100.0%

# attack type: LinfSPSAAttack
# attack kwargs: eps=0.3
#                delta=0.01
#                lr=0.01
#                nb_iter=1000
#                nb_sample=128
#                max_batch_size=64
#                targeted=False
#                loss_fn=None
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 100 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 100.0%
# attack success rate: 10.0%

# attack type: LinfSPSAAttack
# attack kwargs: eps=0.3
#                delta=0.01
#                lr=0.01
#                nb_iter=100
#                nb_sample=8192
#                max_batch_size=64
#                targeted=False
#                loss_fn=None
#                clip_min=0.0
#                clip_max=1.0
# data: mnist_test, 100 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 100.0%
# attack success rate: 6.0%



from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import LinfSPSAAttack

from advertorch_examples.benchmark_utils import benchmark_attack_success_rate

batch_size = 10
num_batch = 10
device = "cuda"

lst_attack = [
    (LinfSPSAAttack, dict(
        eps=0.3, delta=0.01, lr=0.01, nb_iter=1000, nb_sample=128,
        max_batch_size=64, targeted=False,
        loss_fn=None,
        clip_min=0.0, clip_max=1.0)),
    (LinfSPSAAttack, dict(
        eps=0.3, delta=0.01, lr=0.01, nb_iter=100, nb_sample=8192,
        max_batch_size=64, targeted=False,
        loss_fn=None,
        clip_min=0.0, clip_max=1.0)),
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
            model, loader, attack_class, attack_kwargs,
            device=device, num_batch=num_batch))

print(info)
for item in lst_benchmark:
    print(item)
