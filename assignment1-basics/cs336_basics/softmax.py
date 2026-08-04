import torch
from torch import Tensor, nn
from jaxtyping import Float


class SoftMax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        # find the max of v, for stability
        m, _ = in_features.max(dim=self.dim, keepdim=True)
        stablized_in = in_features - m
        exp = torch.exp(stablized_in)
        sumexp = exp.sum(-1, keepdim=True)
        return exp / sumexp
