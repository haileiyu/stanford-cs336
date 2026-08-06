import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
import einops
from einops import einsum
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # if rope params are supplied
        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len)
        else:
            self.rope = None

    def forward(
        self,
        q_proj_weight: Float[Tensor, " d_model d_model"],
        k_proj_weight: Float[Tensor, " d_model d_model"],
        v_proj_weight: Float[Tensor, " d_model d_model"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        q_rearranged = einops.rearrange(q_proj_weight, "(a1 a2) b -> a1 a2 b", a1=self.num_heads)
        k_rearranged = einops.rearrange(k_proj_weight, "(a1 a2) b -> a1 a2 b", a1=self.num_heads)
        v_rearranged = einops.rearrange(v_proj_weight, "(a1 a2) b -> a1 a2 b", a1=self.num_heads)

        # let's do multi head now
        wqx = einsum(q_rearranged, in_features, "num_heads d_head d_model, ... seq d_model -> ... num_heads seq d_head")
        wkx = einsum(k_rearranged, in_features, "num_heads d_head d_model, ... seq d_model -> ... num_heads seq d_head")
        wvx = einsum(v_rearranged, in_features, "num_heads d_head d_model, ... seq d_model -> ... num_heads seq d_head")

        # if token_positions is supplied, we should rope the q and k tensors.
        if self.rope is not None:
            if token_positions is None:
                sequence_length = in_features.shape[-2]
                token_positions = torch.arange(sequence_length, device=in_features.device)
            wqx = self.rope.forward(wqx, token_positions)
            wkx = self.rope.forward(wkx, token_positions)

        d_query = wqx.shape[-2]
        d_key = wkx.shape[-2]
        mask = torch.tril(torch.ones(d_query, d_key)).bool()
        a = scaled_dot_product_attention(wqx, wkx, wvx, mask)  # ... num_heads seq d_head
        # collapse
        a_concat = einops.rearrange(a, "... num_heads seq d_head -> ... seq (num_heads d_head)")  # ... seq d_model
        return einsum(o_proj_weight, a_concat, "a b, ... seq b -> ... seq a")
