from collections.abc import Iterable
import torch
import math


eps = 1e-6


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    # store the input parameters to a list, since it's an Iterable that promises to be iterated once, but
    # we're iterating twice here. if the input is a generator (yield), the second loop would get nothing.
    parameters_list = list(parameters)
    square_sum = 0
    for p in parameters_list:
        if p.grad is not None:
            square_sum += (p.grad**2).sum()

    g_2 = math.sqrt(square_sum)
    if g_2 < max_l2_norm:
        return

    scale_factor = max_l2_norm / (g_2 + eps)

    for p in parameters_list:
        if p.grad is not None:
            p.grad *= scale_factor
