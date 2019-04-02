# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn


class Processor(nn.Module):
    """
    Processor
    """
    def __init__(self):
        super(Processor, self).__init__()

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'EmptyDefense (Identity)'
