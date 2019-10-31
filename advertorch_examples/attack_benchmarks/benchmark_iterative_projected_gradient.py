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

from advertorch_examples.benchmark_utils import benchmark_robust_accuracy
# TODO: from advertorch_examples.benchmark_utils import benchmark_margin

batch_size = 100
device = "cuda"


lst_attack = [
    (LinfPGDAttack, {}),
]
# TODO: add dictionary values

mnist_clntrained_model = get_mnist_lenet5_clntrained()
mnist_clntrained_model.to(device)
mnist_advtrained_model = get_mnist_lenet5_advtrained()
mnist_advtrained_model.to(device)

mnist_test_loader = get_mnist_test_loader(batch_size=batch_size)

lst_setting = [
    (mnist_clntrained_model, mnist_test_loader),
    (mnist_advtrained_model, mnist_test_loader),
]


info = get_benchmark_sys_info()


lst_benchmark = []

for model, loader in lst_setting:
    for attack_class, attack_kwargs in lst_attack:
        lst_benchmark.append(benchmark_robust_accuracy(
            model, loader, attack_class, attack_kwargs))

print(info)
for item in lst_benchmark:
    print(item)
