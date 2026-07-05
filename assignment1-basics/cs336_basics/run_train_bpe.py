import os
import sys
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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

    with open(input_path, "r") as content_file:
        content = content_file.read()
        print(content)

    word_to_count: dict[str, int] = {}
    for m in re.finditer(PAT, content):
        word = m.group()
        word_to_count[word] = word_to_count.get(word, 0) + 1
    print(word_to_count)

    # 3. in a loop:
    #      initial vocabulary is 256 bytes and a textstopper
    #      create a map {pretoken, {vocabulary, count}}
    #      fill the map -- for each word, try the different combinations, and put them into the map
    #      merge the results into {vocabulary, count}
    #      select the highest frequency pair in this round
    #      add that token into the vocabulary
    #      repeat
    vocab_list: list[bytes] = [bytes([i]) for i in range(256)]
    for t in special_tokens:
        vocab_list.append(to_bytes(t))

    vocab = set(vocab_list)
    print(vocab)

    pair_to_count = Counter()
    for w in word_to_count:
        pair_count = get_vocab_pair_count_in_word(w, vocab)
        for p in pair_count:
            pair_to_count[p] += word_to_count[w] * pair_count[p]

    print("> pair_to_count")
    print(pair_to_count)

    # find the highest frequency one

    most_frequent_pair = get_most_frequent_pair(pair_to_count)

    print("> most_frequent_pair")
    print(most_frequent_pair)

    return {}, []


def get_most_frequent_pair(pair_to_count: dict[bytes, int]) -> bytes:
    highest_count = 0
    most_common_pairs: list[bytes] = []
    for pair in pair_to_count:
        if pair_to_count[pair] > highest_count:
            highest_count = pair_to_count[pair]
            most_common_pairs = [pair]
        elif pair_to_count[pair] == highest_count:
            most_common_pairs.append(pair)

    print(most_common_pairs)

    if len(most_common_pairs) == 1:
        return most_common_pairs[0]
    elif len(most_common_pairs) > 1:
        most_common_pairs.sort(reverse=True)
        return most_common_pairs[0]
    else:
        raise RuntimeError


def to_bytes(w: str) -> bytes:
    return w.encode("utf-8")


def get_vocab_pair_count_in_word(word: str, vocab: set[bytes]) -> dict[bytes, int]:
    """
    this is the minimal test example for the "find pair" algorithm.
    """
    pair_to_count = Counter()

    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            curr = to_bytes(word[i:j])
            print(curr)
            if not curr in vocab:
                continue

            for k in range(j + 1, len(word) + 1):
                next = to_bytes(word[j:k])
                if not next in vocab:
                    continue
                pair = curr + next
                pair_to_count[pair] += 1

    return pair_to_count


if __name__ == "__main__":
    # get_vocab_pair_count_in_word('abba', {b'a', b'b'})
    run_train_bpe(sys.argv[1], 500, ["<|endoftext|>"])
