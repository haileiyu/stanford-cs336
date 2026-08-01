import math
import torch
from torch import nn, Tensor


def init_weights(d1: int, d2: int, device=None, dtype=None) -> Tensor:
    std = math.sqrt(2 / (d1 + d2))
    return nn.init.trunc_normal_(torch.zeros(d1, d2, device=device, dtype=dtype), 0, std, -3 * std, 3 * std)
