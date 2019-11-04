# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn

from advertorch.attacks import LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack
from advertorch_examples.utils import get_cifar10_test_loader

from madry_et_al_utils import get_madry_et_al_tf_model

model = get_madry_et_al_tf_model("CIFAR10")
loader = get_cifar10_test_loader(batch_size=100)
adversary = LinfPGDAttack(
    model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
    nb_iter=20, eps_iter=2. / 255, rand_init=False, clip_min=0.0, clip_max=1.0,
    targeted=False)

label, pred, advpred, _ = multiple_mini_batch_attack(
    adversary, loader, device="cuda")

print("Accuracy: {:.2f}%, Robust Accuracy: {:.2f}%".format(
    100. * (label == pred).sum().item() / len(label),
    100. * (label == advpred).sum().item() / len(label)))

# Accuracy: 87.14%, Robust Accuracy: 45.69%
