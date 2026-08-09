import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
import einops
from einops import einsum
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding


# flop:
# 3 x 2 x ... x num_heads x d_k x d_model x seq
# + (4 x d_k + 5) x ... x num_heads x seq x seq
# + 2 x d_model x d_model x ... x seq
# also, + 2 x 3 x ... x num_heads x seq x d_k if rope is enabled
#
# = 6 x ... x seq x d_model^2
# + (4 x d_k + 5) x ... x num_heads x seq^2
# + 2 x ... x seq x d_model^2
# also, + 6 x ... x seq x d_model if rope is enabled
#
# = 8 x ... x seq x d_model^2
# + (4 x d_k + 5) x ... x num_heads x seq^2
# also, + 6 x ... x seq x d_model if rope is enabled
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
        # flop: for each line: 2 x ... x num_heads x d_k x d_model x seq
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
            # flop: for each line: 3 x ... x num_heads x seq x d_k
            wqx = self.rope.forward(wqx, token_positions)
            wkx = self.rope.forward(wkx, token_positions)

        d_query = wqx.shape[-2]
        d_key = wkx.shape[-2]
        # memory: d_query x d_key
        # flop: 0
        mask = torch.tril(torch.ones(d_query, d_key, device=device)).bool()

        # memory: todo
        # flop: # 2 x ... x queries x keys x d_k +
        #         5 x ... x queries x keys +
        #         2 x ... x queries x keys x d_v
        # where queries is seq, keys is seq, d_k == d_v, ... is ... x num_heads
        # so the result is:
        # 4 x ... x num_heads x seq x seq x d_k + 5 x ... x num_heads x seq x seq
        # = (4 x d_k + 5) x ... x num_heads x seq x seq
        a = scaled_dot_product_attention(wqx, wkx, wvx, mask)  # shape: (..., num_heads, seq d_k)

        # collapse
        # flop: 0
        # memory: ... x num_heads x seq x d_k (since this moves axis across another)
        a_concat = einops.rearrange(a, "... num_heads seq d_k -> ... seq (num_heads d_k)")  # shape: (..., seq, d_model)

        # flop: 2 x d_model x d_model x ... x seq
        # memory: todo
        r = einsum(o_proj_weight, a_concat, "a b, ... seq b -> ... seq a")
        return r
