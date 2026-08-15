import torch
import einops
from einops import einsum
import numpy
from cs336_basics.adamw import AdamW
from cs336_basics.transformer import TransformerLM
from cs336_basics.data_loading import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import lr_cosine_schedule


batch_size = 32
context_length = 256
vocab_size = 10000
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16

# gradient clipping
max_l2_norm = 1.0

# learning rate schedule
num_iterations = 1000
max_lr = 1e-3
min_lr = 1e-4
warmup_iters = 50
cosine_cycle_iters = num_iterations

device = "mps"
path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_ids.npy"


def training_loop(num_iterations: int):

    dataset = numpy.load(path, mmap_mode="r")

    tlm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)
    optimizer = AdamW(params=tlm.parameters())
    for t in range(num_iterations):
        lr = lr_cosine_schedule(t, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        in_indices = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)
        input = in_indices[0]
        flattened_expected_out = einops.rearrange(in_indices[1], "a b -> (a b)")
        out = tlm(input)
        flattened_out = einops.rearrange(out, "a b c -> (a b) c")

        optimizer.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = cross_entropy(flattened_out, flattened_expected_out)
        print(loss.item())
        loss.backward()  # Run backward pass, which computes gradients.

        gradient_clipping(tlm.parameters(), max_l2_norm)

        optimizer.step()  # Run optimizer step.


if __name__ == "__main__":
    training_loop(num_iterations)
