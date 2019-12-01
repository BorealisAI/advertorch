import torch.nn as nn

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
