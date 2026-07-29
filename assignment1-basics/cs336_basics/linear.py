from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
import torch
from einops import einsum, rearrange
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        init_weights = nn.init.trunc_normal_(torch.zeros(out_features, in_features), 0, std, -3 * std, 3 * std)
        self.weights: nn.Parameter = nn.Parameter(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        print(x.shape)
        assert x.shape[-1] == self.weights.data.shape[1]
        out = einsum(x, self.weights, "... b, c b -> ... c")
        return out


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    l = Linear(d_in, d_out)
    l.load_state_dict({"weights": weights})
    return l.forward(in_features)


if __name__ == "__main__":
    print("linear")
