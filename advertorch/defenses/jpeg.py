# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from advertorch.functional import JPEGEncodingDecoding

from .base import Processor


class JPEGFilter(Processor):
    """
    JPEG Filter.

    :param quality: quality of the output.
    """
    def __init__(self, quality=75):
        super(JPEGFilter, self).__init__()
        self.quality = quality

    def forward(self, x):
        return JPEGEncodingDecoding.apply(x, self.quality)
