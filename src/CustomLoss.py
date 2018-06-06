#!/usr/bin/env python3

import torch
# import torch.nn as nn
from torch.nn.modules.loss import MSELoss, _assert_no_grad

class WeightedMSELoss(MSELoss):
    """Creates a criterion that measures the weighted mean squared error between
    `n` elements in the input `x` and target `y`.

    The sum operation still operates over all the elements, and divides by `n`.
    The division by `n` can be avoided if one sets :attr:`size_average` to ``False``.
    To get a batch of losses, a loss per batch element, set `reduce` to
    ``False``. These losses are not averaged and are not affected by
    `size_average`.
    Args:
        size_average (bool, optional): By default, the losses are averaged
                   over each loss element in the batch. Note that for some losses, there
                   multiple elements per sample. If the field :attr:`size_average` is set to
                   ``False``, the losses are instead summed for each minibatch. Ignored
                   when reduce is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
                   over observations for each minibatch, or summed, depending on
                   size_average. When reduce is ``False``, returns a loss per input/target
                   element instead and ignores size_average. Default: ``True``
        Shape:
                - Input: :math:`(N, *)` where `*` means, any number of additional
                  dimensions
                - Target: :math:`(N, *)`, same shape as the input
    """
    def __init__(self, size_average=True, reduce=True):
        super(MSELoss, self).__init__(size_average, reduce)

    def forward(self, input, target, weight_matrix,
                size_average=True,
                reduce = True):
        _assert_no_grad(target)

        l = torch.mm((input - target)**2, weight_matrix)

        if not self.reduce:
            return l
        return torch.mean(l) if self.size_average else torch.sum(l)
