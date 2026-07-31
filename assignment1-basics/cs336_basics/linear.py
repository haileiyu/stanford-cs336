from torch import Tensor, nn
import torch
from einops import einsum
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        init_weights = nn.init.trunc_normal_(
            torch.zeros(out_features, in_features, device=device, dtype=dtype), 0, std, -3 * std, 3 * std
        )
        self.weights: nn.Parameter = nn.Parameter(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        out = einsum(x, self.weights, "... b, c b -> ... c")
        return out


if __name__ == "__main__":
    print("linear")
