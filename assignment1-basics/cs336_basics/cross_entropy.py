import torch
from torch import Tensor
from jaxtyping import Int, Float


def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # first, for numeric stability, subtract the max of the row
    m, _ = inputs.max(dim=-1, keepdim=True)
    stablized_in = inputs - m

    # second, do the simplified cross entropy formula
    e = torch.exp(stablized_in)
    sumexp = e.sum(-1, keepdim=True)
    logsumexp = torch.log(sumexp)  # shape should be (batch_size,)

    batch_size = inputs.shape[0]
    i = stablized_in[torch.arange(batch_size), targets]

    t = logsumexp - i
    return t.mean()
