import torch

from torch import nn, Tensor
from cs336_basics.init_weights import init_weights
from cs336_basics.linear import Linear
from jaxtyping import Bool, Float, Int
from einops import einsum


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, in_features: Float[Tensor, " ... d_model"]):
        w1x = self.w1(in_features)

        silu_w1x = silu(w1x)

        w3x = self.w3(in_features)

        return self.w2(silu_w1x * w3x)


def silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)
