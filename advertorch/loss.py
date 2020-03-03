import torch
from torch.nn.modules.loss import _Loss
from advertorch.utils import clamp


class ZeroOneLoss(_Loss):
    """Zero-One Loss"""

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean'):
        super(ZeroOneLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return logit_margin_loss(input, target, reduction=self.reduction)



class LogitMarginLoss(_Loss):
    """Logit Margin Loss"""

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean', offset=0.):
        super(LogitMarginLoss, self).__init__(size_average, reduce, reduction)
        self.offset = offset

    def forward(self, input, target):
        return logit_margin_loss(
            input, target, reduction=self.reduction, offset=self.offset)


class CWLoss(_Loss):
    """CW Loss"""
    # TODO: combine with the CWLoss in advertorch.utils

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean'):
        super(CWLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return cw_loss(input, target, reduction=self.reduction)


class SoftLogitMarginLoss(_Loss):
    """Soft Logit Margin Loss"""

    def __init__(self, size_average=None, reduce=None,
                 reduction='elementwise_mean', offset=0.):
        super(SoftLogitMarginLoss, self).__init__(
            size_average, reduce, reduction)
        self.offset = offset

    def forward(self, logits, targets):
        return soft_logit_margin_loss(
            logits, targets, reduction=self.reduction, offset=self.offset)


def zero_one_loss(input, target, reduction='elementwise_mean'):
    loss = (input != target)
    return _reduce_loss(loss, reduction)


def elementwise_margin(logits, label):
    batch_size = logits.size(0)
    topval, topidx = logits.topk(2, dim=1)
    maxelse = ((label != topidx[:, 0]).float() * topval[:, 0]
               + (label == topidx[:, 0]).float() * topval[:, 1])
    return maxelse - logits[torch.arange(batch_size), label]


def logit_margin_loss(input, target, reduction='elementwise_mean', offset=0.):
    loss = elementwise_margin(input, target)
    return _reduce_loss(loss, reduction) + offset


def cw_loss(input, target, reduction='elementwise_mean'):
    loss = clamp(elementwise_margin(input, target) + 50, 0.)
    return _reduce_loss(loss, reduction)


def _reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(reduction + " is not valid")


def soft_logit_margin_loss(
        logits, targets, reduction='elementwise_mean', offset=0.):
    batch_size = logits.size(0)
    num_class = logits.size(1)
    mask = torch.ones_like(logits).byte()
    # TODO: need to cover different versions of torch
    # mask = torch.ones_like(logits).bool()
    mask[torch.arange(batch_size), targets] = 0
    logits_true_label = logits[torch.arange(batch_size), targets]
    logits_other_label = logits[mask].reshape(batch_size, num_class - 1)
    loss = torch.logsumexp(logits_other_label, dim=1) - logits_true_label
    return _reduce_loss(loss, reduction) + offset
