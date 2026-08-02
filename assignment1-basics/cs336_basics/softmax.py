import torch
from torch import Tensor, nn
from jaxtyping import Float


class SoftMax(nn.Module):
    def __init__(self, dim: int):
        self.dim = dim

    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        # find the max of v, for stability
        m, _ = in_features.max(dim=-1, keepdim=True)
        stablized_in = in_features - m
        exp = torch.exp(stablized_in)
        sumexp = exp.sum(-1, keepdim=True)
        return exp / sumexp
