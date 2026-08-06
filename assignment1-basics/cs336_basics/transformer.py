import torch
from torch import nn, Tensor
from jaxtyping import Float
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int | None = None, theta: float | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.rmsnorm = RMSNorm(d_model)
        self.multi_head_self_attention = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.swiglu = SwiGLU(d_model, d_ff)

    def forward(
        self,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        # first, the attention
        r = self.rmsnorm
        r.load_state_dict({"weights": weights["ln1.weight"]})
        m = self.multi_head_self_attention(
            weights["attn.q_proj.weight"],
            weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],
            weights["attn.output_proj.weight"],
            r(in_features),
        )

        y = in_features + m

        # second, the feed forward
        r.load_state_dict({"weights": weights["ln2.weight"]})
        ry = r(y)
        s = self.swiglu
        s.load_state_dict(
            {
                "w1_weight": weights["ffn.w1.weight"],
                "w2_weight": weights["ffn.w2.weight"],
                "w3_weight": weights["ffn.w3.weight"],
            }
        )
        return y + s(ry)
