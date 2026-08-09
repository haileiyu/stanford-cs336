import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
import einops
from einops import einsum
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding


# flop:
# 3 * 2 * ... * num_heads * d_k * d_model * seq
# + (4 * d_k + 5) * ... * num_heads * seq * seq
# + 2 * d_model * d_model * ... * seq
# also, + 2 * 3 * ... * num_heads * seq * d_k if rope is enabled
#
# = 6 * ... * seq * d_model^2
# + (4 * d_k + 5) * ... * num_heads * seq^2
# + 2 * ... * seq * d_model^2
# also, + 6 * ... * seq * d_model if rope is enabled
#
# = 8 * ... * seq * d_model^2
# + (4 * d_k + 5) * ... * num_heads * seq^2
# also, + 6 * ... * seq * d_model if rope is enabled
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
        in_features: Float[Tensor, " ... seq d_model"],
        token_positions: Int[Tensor, " ... seq"] | None = None,
    ) -> Float[Tensor, " ... seq d_model"]:
        # rearranges, if not moving an axis past another one, is a ~free operation (no memory movements, no arithmetic)
        # flop: 0
        q_rearranged = einops.rearrange(
            q_proj_weight, "(a1 a2) b -> a1 a2 b", a1=self.num_heads
        )  # shape: (num_heads, d_k, d_model)
        k_rearranged = einops.rearrange(k_proj_weight, "(a1 a2) b -> a1 a2 b", a1=self.num_heads)
        v_rearranged = einops.rearrange(v_proj_weight, "(a1 a2) b -> a1 a2 b", a1=self.num_heads)

        # let's do multi head now
        # flop: for each line: 2 * ... * num_heads * d_k * d_model * seq
        wqx = einsum(
            q_rearranged, in_features, "num_heads d_k d_model, ... seq d_model -> ... num_heads seq d_k"
        )  # shape is (... num_heads, seq, d_k)
        wkx = einsum(k_rearranged, in_features, "num_heads d_k d_model, ... seq d_model -> ... num_heads seq d_k")
        wvx = einsum(v_rearranged, in_features, "num_heads d_k d_model, ... seq d_model -> ... num_heads seq d_k")

        device = in_features.device
        # if token_positions is supplied, we should rope the q and k tensors.
        if self.rope is not None:
            if token_positions is None:
                seq = in_features.shape[-2]
                token_positions = torch.arange(seq, device=device)
            # flop: for each line: 3 * ... * num_heads * seq * d_k
            wqx = self.rope.forward(wqx, token_positions)
            wkx = self.rope.forward(wkx, token_positions)

        d_query = wqx.shape[-2]
        d_key = wkx.shape[-2]
        # memory: d_query * d_key
        # flop: 0
        mask = torch.tril(torch.ones(d_query, d_key, device=device)).bool()

        # memory: todo
        # flop: # 2 * ... * queries * keys * d_k +
        #         5 * ... * queries * keys +
        #         2 * ... * queries * keys * d_v
        # where queries is seq, keys is seq, d_k == d_v, ... is ... * num_heads
        # so the result is:
        # 4 * ... * num_heads * seq * seq * d_k + 5 * ... * num_heads * seq * seq
        # = (4 * d_k + 5) * ... * num_heads * seq * seq
        a = scaled_dot_product_attention(wqx, wkx, wvx, mask)  # shape: (..., num_heads, seq d_k)

        # collapse
        # flop: 0
        # memory: ... * num_heads * seq * d_k (since this moves axis across another)
        a_concat = einops.rearrange(a, "... num_heads seq d_k -> ... seq (num_heads d_k)")  # shape: (..., seq, d_model)

        # flop: 2 * d_model * d_model * ... * seq
        # memory: todo
        r = einsum(o_proj_weight, a_concat, "a b, ... seq b -> ... seq a")
        return r
