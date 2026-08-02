import torch
from torch import Tensor, nn
from einops import einsum


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

    def forward(self, q: Tensor, token_positions: Tensor) -> Tensor:
        """input shapes: q: Float[Tensor, " ... sequence_length d_k"]
        token_positions: Int[Tensor, " ... sequence_length"]"""
        thetas = self.create_theta(token_positions)
        c, s = torch.cos(thetas), torch.sin(thetas)
        q1, q2 = q[..., 0::2], q[..., 1::2]
        r = torch.stack([c * q1 - s * q2, s * q1 + c * q2], dim=-1).flatten(-2)
        return r

    def create_theta(self, token_positions: Tensor):
        k = (torch.arange(self.d_k / 2)) * 2 / self.d_k
        # [:, None] adds a trailing dimension for the broadcast division to work
        return token_positions[:, None] / torch.pow(self.theta, k)
