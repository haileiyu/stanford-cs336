import torch
from torch import Tensor, nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        init_embedding = nn.init.trunc_normal_(
            torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype), 0, 1, -3, 3
        )
        self.embedding: nn.Parameter = nn.Parameter(init_embedding)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding[token_ids]


if __name__ == "__main__":
    print("hello")
