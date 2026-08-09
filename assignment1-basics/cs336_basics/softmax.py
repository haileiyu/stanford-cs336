import torch
from torch import Tensor, nn
from jaxtyping import Float


class SoftMax(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    # flop: 4 * ... (element count)
    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        # find the max of v, for stability
        m, _ = in_features.max(dim=self.dim, keepdim=True)

        # flop: ...
        stablized_in = in_features - m

        # flop: ...
        exp = torch.exp(stablized_in)

        # flop: ...
        sumexp = exp.sum(self.dim, keepdim=True)

        # flop: ...
        return exp / sumexp
