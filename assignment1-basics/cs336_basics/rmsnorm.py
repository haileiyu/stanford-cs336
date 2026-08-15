import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        # todo: replace the weight initialization with init_weights
        self.d_model = d_model
        init_weight = torch.ones(d_model, device=device, dtype=dtype)
        self.weight: nn.Parameter = nn.Parameter(init_weight)
        self.eps = eps

    # flop: 3 * ... + 2 * ... * d_model + 2 * ... * d_model
    # = 4 * ... * d_model + 3 * ...
    def forward(self, x: Tensor) -> Tensor:
        """x: Float[Tensor, " ... d_model]"""
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # flop: ... * d_model + ... * d_model = 2 * ... * d_model
        suma = (x * x).sum(-1, keepdim=True)  # shape: (...,)
        # flop: ...
        suma /= self.d_model  # shape: (...,)
        # flop: ...
        suma += self.eps
        # flop: ...
        suma = torch.sqrt(suma)
        # flop: ... * d_model + ... * d_model = 2 * ... * d_model
        result = x * self.weight / suma

        return result.to(in_dtype)
