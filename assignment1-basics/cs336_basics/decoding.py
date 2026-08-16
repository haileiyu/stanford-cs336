from cs336_basics.transformer import TransformerLM
from cs336_basics.training_loop import (
    d_model,
    vocab_size,
    context_length,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    device,
)
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.adamw import AdamW
from cs336_basics.data_loading import load_checkpoint
import torch



endoftext_str = "<|endoftext|>"

class Decoder:
    def __init__(self):
        self.tlm = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        ).to(device)
        self.temperature = 0.8

    def generate(self, input):
        raise NotImplementedError


if __name__ == "__main__":
    prompt = "a fat cat is"

    # tokenize
    vocab_path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_vocab.pkl"
    merges_path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_merges.pkl"
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, [endoftext_str])

    checkpoint_path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_checkpoint.cpt"

    # encoding
    token_indexes = tokenizer.encode(prompt)
    
    endoftext_index = tokenizer.inverse_vocab[endoftext_str.encode("utf-8")]

    # forward
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

    checkpointed_iter = load_checkpoint(checkpoint_path, tlm, optimizer)

    num_iters = 100

    print(prompt)

    # torch.no_grad would save memory by turning off the grad (for back propagation)
    with torch.no_grad():

        for i in range(num_iters):
            # truncate to context_length
            context_window = token_indexes[-context_length :]

            # convert to a tensor, flatten, and feed into transformer
            out = tlm(torch.tensor(context_window, dtype=torch.long, device=device).unsqueeze(0))

            # pick the most likely token, and append to the list, and loop
            index = out[0, -1, :].argmax().item()
            
            # break if end of text
            if index == endoftext_index:
                print("end")
                break
            
            token_indexes.append(index)
            print(tokenizer.decode([index]), end="", flush=True)

    print()
    print(token_indexes)
    print(tokenizer.decode(token_indexes))
