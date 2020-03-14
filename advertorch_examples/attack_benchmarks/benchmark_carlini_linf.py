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


from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info

from advertorch.attacks import CarliniWagnerLinfAttack

from advertorch_examples.benchmark_utils import benchmark_margin

batch_size = 100
device = "cuda"

lst_attack = [
    (CarliniWagnerLinfAttack, dict(num_classes=10,
                                   max_iterations=500,
                                   max_const=20,
                                   return_best=False)),
    (CarliniWagnerLinfAttack, dict(num_classes=10,
                                   max_iterations=500,
                                   max_const=20,
                                   return_best=True)),
    (CarliniWagnerLinfAttack, dict(num_classes=10,
                                   max_iterations=500,
                                   max_const=0.1,
                                   return_best=False)),
    (CarliniWagnerLinfAttack, dict(num_classes=10,
                                   max_iterations=500,
                                   max_const=0.1,
                                   return_best=True)),
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
            model, loader, attack_class, attack_kwargs, "inf", device="cuda"))

print(info)
for item in lst_benchmark:
    print(item)
