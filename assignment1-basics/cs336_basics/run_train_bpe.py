import os
import sys

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # in the first version, i'm gonna do it in the simplest way possible.
    # the algorithm can be broken down into:
    # 1. load the entire file into memory. future optimzations: scan each line and process each line.
    # 2. pre-tokenization: split the file by whitespaces. store the results as a {word, count} map.
    # 3. in a loop:
    #      initial vocabulary is 256 bytes and a textstopper
    #      create a map {word, []{vocabulary, count}}
    #      merge the results into {token, count}
    #      select the highest frequency token in this round
    #      add that token into the vocabulary
    #      repeat
    # 4. finally, print out the tokens
    # note:
    # - should probably start with a smaller text file
    # - should print the output of each step

    # step 1. pretokenization
    # Source - https://stackoverflow.com/a/7409814

    with open(input_path, 'r') as content_file:
        content = content_file.read()
        print(content)
    words = content.split(' ')

    word_count = {}
    for w in words:
        if w in word_count:
            word_count[w] = word_count[w] + 1
        else:
            word_count[w] = 1
    print(word_count)

    return {}, []
    



if __name__ == "__main__":
    run_train_bpe(sys.argv[1], 500, [])