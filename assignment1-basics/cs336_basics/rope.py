import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # let's cache the sins and coses
        t = self._create_theta(torch.arange(max_seq_len, device=device))
        c, s = torch.cos(t), torch.sin(t)
        self.register_buffer("cos_cached", c, persistent=False)
        self.register_buffer("sin_cached", s, persistent=False)

    def forward(self, q: Tensor, token_positions: Tensor) -> Tensor:
        """input shapes: q: Float[Tensor, " ... sequence_length d_k"]
        token_positions: Int[Tensor, " ... sequence_length"]"""
        c = self.cos_cached[token_positions]
        s = self.sin_cached[token_positions]
        q1, q2 = q[..., 0::2], q[..., 1::2]
        r = torch.stack([c * q1 - s * q2, s * q1 + c * q2], dim=-1).flatten(-2)
        return r

    def _create_theta(self, token_positions: Tensor):
        # d_k / 2 is a float. use // to avoid potential landmines in the future.
        k = (torch.arange(self.d_k // 2, device=self.device)) * 2 / self.d_k
        # [:, None] adds a trailing dimension for the broadcast division to work
        return token_positions[:, None] / torch.pow(self.theta, k)
