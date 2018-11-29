# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn

from advertorch.attacks import LinfPGDAttack
from advertorch.attacks.utils import AttackConfig


class PGDLinfMadryTrainMnist(AttackConfig):
    AttackClass = LinfPGDAttack
    eps = 0.3
    eps_iter = 0.01
    nb_iter = 40
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    rand_init = True
    clip_min = 0.0
    clip_max = 1.0


class PGDLinfMadryTestMnist(PGDLinfMadryTrainMnist):
    # only modify the entry that is changed from PGDLinfMadryTrainMnist
    nb_iter = 100


if __name__ == '__main__':
    from advertorch.test_utils import LeNet5

    model = LeNet5()

    train_adversary = PGDLinfMadryTrainMnist()(model)
    test_adversary = PGDLinfMadryTestMnist()(model)
