import einops
import numpy
from cs336_basics.adamw import AdamW
from cs336_basics.transformer import TransformerLM
from cs336_basics.data_loading import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.learning_rate_schedule import lr_cosine_schedule
from cs336_basics.data_loading import save_checkpoint, load_checkpoint
import wandb


# batch_size = 32
context_length = 256
vocab_size = 10000
d_model = 512
d_ff = 1344
rope_theta = 10000
num_layers = 4
num_heads = 16

# gradient clipping
max_l2_norm = 1.0

# training budget: batch_size * num_iterations * context_length ~= token_budget
token_budget = 40_000_000

# learning rate schedule
# max_lr = 1e-4
# min_lr = max_lr / 10
warmup_frac = 0.01  # was warmup_iters = 50, i.e. 1% of the old 5000 steps

checkpoint_iters = 500
checkpoint_dir = "/Users/admin/cs336/assignment1-basics"

device = "mps"
training_data = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_ids.npy"


def training_loop(num_iterations: int, max_lr: float, batch_size: int, should_load_checkpoint: bool = False):

    # these all follow num_iterations, so they have to be derived per run
    cosine_cycle_iters = num_iterations
    warmup_iters = max(10, int(warmup_frac * num_iterations))
    checkpoint_file = f"{checkpoint_dir}/tiny_stories_10000_bs{batch_size}.cpt"

    dataset = numpy.load(training_data, mmap_mode="r")

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

    checkpointed_iter = -1  # since we need to +1 later
    if should_load_checkpoint:
        checkpointed_iter = load_checkpoint(checkpoint_file, tlm, optimizer)
        print("loaded checkpoint at iteration", checkpointed_iter)
        
    wandb.init(
        project="cs336-a1",
        name=f"bs{batch_size}_lr{max_lr:.1e}",
        config={
            "lr": max_lr,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
            "tokens": batch_size * num_iterations * context_length,
        },
    )

    for iter in range(checkpointed_iter + 1, num_iterations):
        lr = lr_cosine_schedule(iter, max_lr, max_lr/10, warmup_iters, cosine_cycle_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr

        in_indices = get_batch(dataset, batch_size=batch_size, context_length=context_length, device=device)
        input = in_indices[0]
        flattened_expected_out = einops.rearrange(in_indices[1], "a b -> (a b)")
        out = tlm(input)
        flattened_out = einops.rearrange(out, "a b c -> (a b) c")

        optimizer.zero_grad()  # reset the gradients for all learnable parameters
        loss = cross_entropy(flattened_out, flattened_expected_out)
        # print(loss.item())
        wandb.log({"loss": loss, "lr": lr, "tokens": (iter + 1) * batch_size * context_length}, step=iter)
        loss.backward()  # run backward pass, which computes gradients

        gradient_clipping(tlm.parameters(), max_l2_norm)

        optimizer.step()

        # check point
        if iter % checkpoint_iters == 0 and iter > checkpoint_iters:
            print("saving checkpoint at iteration", iter)
            save_checkpoint(tlm, optimizer, iter, checkpoint_file)

    # final save
    print("finally, saving the training")
    save_checkpoint(tlm, optimizer, num_iterations - 1, checkpoint_file)
    wandb.finish()


if __name__ == "__main__":
    for batch_size in [16, 32, 64, 128]:
        num_iterations = token_budget // (batch_size * context_length)
        lr = 1e-3 * (batch_size / 32) ** 0.5
        print(f"batch_size={batch_size} num_iterations={num_iterations} lr={lr:.2e}")
        training_loop(num_iterations, lr, batch_size, False)
