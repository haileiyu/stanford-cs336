import torch
from torch import nn, Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        # todo: replace the weight initialization with init_weights
        self.d_model = d_model
        init_weights = torch.ones(d_model, device=device, dtype=dtype)
        self.weights: nn.Parameter = nn.Parameter(init_weights)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        suma = (x * x).sum(-1, keepdim=True)
        suma /= self.d_model
        suma += self.eps
        suma = torch.sqrt(suma)
        result = x * self.weights / suma

        return result.to(in_dtype)
