import torch
from torch import Tensor
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
from cs336_basics.softmax import SoftMax


endoftext_str = "<|endoftext|>"


class Decoder:
    def __init__(self, vocab_path: str, merges_path: str, temperature=0.8, top_p_threshold=0.9):
        self.temperature = temperature
        self.top_p_threshold = top_p_threshold
        self.tokenizer = Tokenizer.from_files(vocab_path, merges_path, [endoftext_str])
        self.endoftext_index = self.tokenizer.inverse_vocab[endoftext_str.encode("utf-8")]

        self.transformer = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
        ).to(device)
        self.optimizer = AdamW(params=self.transformer.parameters())
        self.soft_max = SoftMax(0)

    def top_p(self, input: Tensor) -> int:
        sorted, orig_positions = torch.sort(input, descending=True)

        # find the first n elements that sum to p
        sum = 0.0
        i_p = 0
        for i in range(0, len(sorted)):
            sum += sorted[i]
            if sum >= self.top_p_threshold:
                i_p = i
                break

        sorted = sorted[0 : i_p + 1]
        selected_probability = int(torch.multinomial(sorted, 1).item())

        return int(orig_positions[selected_probability].item())

    def decode(self, prompt: str, num_iters: int, checkpoint_path):
        checkpointed_iter = load_checkpoint(checkpoint_path, self.transformer, self.optimizer)
        token_indexes = self.tokenizer.encode(prompt)

        # torch.no_grad would save memory by turning off the grad (for back propagation)
        with torch.no_grad():
            for i in range(num_iters):
                # truncate to context_length
                context_window = token_indexes[-context_length:]

                # convert to a tensor, flatten, and feed into transformer
                out = self.transformer(torch.tensor(context_window, dtype=torch.long, device=device).unsqueeze(0))

                # divide by temperature
                sml = self.soft_max(out[0, -1, :] / self.temperature)

                # top-p
                index = self.top_p(sml)

                # break if end of text
                if index == self.endoftext_index:
                    print("~~~end~~~")
                    break

                token_indexes.append(index)
                print(self.tokenizer.decode([index]), end="", flush=True)


if __name__ == "__main__":
    vocab_path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_vocab.pkl"
    merges_path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_merges.pkl"
    checkpoint_path = "/Users/admin/cs336/assignment1-basics/tiny_stories_10000_checkpoint.cpt"
    d = Decoder(vocab_path, merges_path)

    prompt = "holy cow!"
    print(prompt)
    d.decode(prompt, 200, checkpoint_path)
