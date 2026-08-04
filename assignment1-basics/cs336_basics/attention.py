from torch import Tensor
from jaxtyping import Float, Bool
import math
from einops import einsum

from cs336_basics.softmax import SoftMax


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    qkt = einsum(Q, K, "... q dk, ... k dk -> ... q k")
    d_k = Q.shape[-1]
    qkt2 = qkt / math.sqrt(d_k)
    if mask is not None:
        qkt2 = qkt2.masked_fill(~mask, float("-inf"))
    s = SoftMax(-1)  # -1 means to softmax over the last dimension.
    return einsum(s(qkt2), V, "... q k, ... k dv -> ... q dv")
