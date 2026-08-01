from torch import Tensor, nn
from einops import einsum
from cs336_basics.init_weights import init_weights


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weights: nn.Parameter = nn.Parameter(init_weights(out_features, in_features, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        out = einsum(x, self.weights, "... b, c b -> ... c")
        return out


if __name__ == "__main__":
    print("linear")
