from torch import nn, Tensor
from jaxtyping import Float, Int
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_self_attention import MultiheadSelfAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear


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

        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        # first, the attention
        m = self.attn(self.ln1(in_features))

        y = in_features + m

        # second, the feed forward
        ry = self.ln2(y)

        return y + self.ffn(ry)


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

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            tb = TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta)
            self.layers.append(tb)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ):
        # token embedding
        # in_indices has shape (batch seq)
        # the semantics for embedding is: for each element in the in_indices tensor, lookup its value
        # in e. value shape is (d_model,), so the shape of ex is (batch, seq d_model)
        ex = self.token_embeddings(in_indices)  # shape is (batch seq d_model)

        for tb in self.layers:
            ex = tb(ex)  # shape is (batch seq d_model)

        # norm
        nx = self.ln_final(ex)  # shape is (batch seq d_model)

        # linear
        lnx = self.lm_head(nx)  # shape is (batch seq vocab_size)

        # for some reason the test case doesn't expect me to do softmax, though the transformer paper does.
        return lnx
