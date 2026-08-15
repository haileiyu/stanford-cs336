import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        initial_weight = nn.init.trunc_normal_(
            torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype), 0, 1, -3, 3
        )
        self.weight: nn.Parameter = nn.Parameter(initial_weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]
