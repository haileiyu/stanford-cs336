import torch

from torch import nn, Tensor
from cs336_basics.init_weights import init_weights
from jaxtyping import Bool, Float, Int
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1_weight: nn.Parameter = nn.Parameter(init_weights(d_ff, d_model))
        self.w2_weight: nn.Parameter = nn.Parameter(init_weights(d_model, d_ff))
        self.w3_weight: nn.Parameter = nn.Parameter(init_weights(d_ff, d_model))

    def forward(self, in_features: Float[Tensor, " ... d_model"]):
        w1x = einsum(in_features, self.w1_weight, "... b, c b -> ... c")
        silu_w1x = silu(w1x)

        w3x = einsum(in_features, self.w3_weight, "... b, c b -> ... c")

        return einsum(silu_w1x * w3x, self.w2_weight, "... c, b c -> ... b")


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)
