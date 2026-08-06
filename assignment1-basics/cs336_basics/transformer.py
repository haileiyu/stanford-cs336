import torch
from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.softmax import SoftMax


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


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

    def forward(
        self,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ):
        # token embedding
        e = Embedding(self.vocab_size, self.d_model)  # shape is (vocab_size, d_model)
        e.load_state_dict({"embedding": weights["token_embeddings.weight"]})
        # in_indices has shape (batch seq)
        # the semantics for embedding is: for each element in the in_indices tensor, lookup its value
        # in e. value shape is (d_model,), so the shape of ex is (batch, seq d_model)
        ex = e(in_indices)  # shape is (batch seq d_model)

        # num_layers of transformer blocks
        tb = TransformerBlock(
            self.d_model, self.num_heads, self.d_ff, max_seq_len=self.context_length, theta=self.rope_theta
        )
        for i in range(self.num_layers):
            weights_i = {}  # todo: avoid copying
            weights_i["attn.q_proj.weight"] = weights["layers.{num_layers}.attn.q_proj.weight".format(num_layers=i)]
            weights_i["attn.k_proj.weight"] = weights["layers.{num_layers}.attn.k_proj.weight".format(num_layers=i)]
            weights_i["attn.v_proj.weight"] = weights["layers.{num_layers}.attn.v_proj.weight".format(num_layers=i)]
            weights_i["attn.output_proj.weight"] = weights[
                "layers.{num_layers}.attn.output_proj.weight".format(num_layers=i)
            ]
            weights_i["ln1.weight"] = weights["layers.{num_layers}.ln1.weight".format(num_layers=i)]
            weights_i["ln2.weight"] = weights["layers.{num_layers}.ln2.weight".format(num_layers=i)]
            weights_i["ffn.w1.weight"] = weights["layers.{num_layers}.ffn.w1.weight".format(num_layers=i)]
            weights_i["ffn.w2.weight"] = weights["layers.{num_layers}.ffn.w2.weight".format(num_layers=i)]
            weights_i["ffn.w3.weight"] = weights["layers.{num_layers}.ffn.w3.weight".format(num_layers=i)]

            ex = tb(weights_i, ex)  # shape is (batch seq d_model)

        # norm
        n = RMSNorm(self.d_model)
        n.load_state_dict({"weights": weights["ln_final.weight"]})
        nx = n(ex)  # shape is (batch seq d_model)

        # linear
        l = Linear(self.d_model, self.vocab_size)
        l.load_state_dict({"weights": weights["lm_head.weight"]})
        lnx = l(nx)  # shape is (batch seq vocab_size)

        # for some reason the test case doesn't expect me to do softmax, though the transformer paper does.
        # s = SoftMax(-1)
        # return s(lnx) # shape is (batch seq vocab_size)
        return lnx
