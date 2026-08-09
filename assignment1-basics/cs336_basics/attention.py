from torch import Tensor
from jaxtyping import Float, Bool
import math
from einops import einsum

from cs336_basics.softmax import SoftMax


# overall flop:
#  2 x ... x queries x keys x d_k +
#  5 x ... x queries x keys +
#  2 x ... x queries x keys x d_v
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
    # memory: todo
    # flop: 2 x ... x queries x keys x d_k
    qkt = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.shape[-1]

    # flop: ... x queries x keys
    qkt2 = qkt / math.sqrt(d_k)
    if mask is not None:
        # flop: 0, since we're just filling things, not doing math
        qkt2 = qkt2.masked_fill(~mask, float("-inf"))

    s = SoftMax(-1)  # -1 means to softmax over the last dimension.
    # flop: 4 x ... x queries x keys + 2 x ... x queries x keys x d_v
    return einsum(s(qkt2), V, "... queries keys, ... keys d_v -> ... queries d_v")
