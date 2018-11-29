# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

import torch
from torchvision import transforms
from PIL import Image

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


class FloatToIntSqueezing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_int, vmin, vmax):
        # here assuming 0 =< x =< 1
        x = (x - vmin) / (vmax - vmin)
        x = torch.round(x * max_int) / max_int
        return x * (vmax - vmin) + vmin

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", FloatToIntSqueezing)



class JPEGEncodingDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return x.new_tensor(torch.stack(lst_img))

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", JPEGEncodingDecoding)
