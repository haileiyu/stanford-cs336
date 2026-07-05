import os
import sys
import regex as re
import inspect
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_word_tuples_to_count(input_path: str | os.PathLike) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "r") as content_file:
        content = content_file.read()

    word_to_count = Counter()
    for m in re.finditer(PAT, content):
        word = m.group()
        word_to_count[word] += 1
    print_var(word_to_count)

    word_tuples_to_count: dict[tuple[bytes, ...], int] = {}
    for word in word_to_count:
        # split the word into tuples
        # we can use a hack here since the initial vocabulary are all singe bytes
        bytes_tuple = tuple(char.encode("utf-8") for char in word)
        word_tuples_to_count[bytes_tuple] = word_to_count[word]

    return word_tuples_to_count


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

    # pretokenization
    word_tuples_to_count = get_word_tuples_to_count(input_path)
    print_var(word_tuples_to_count)

    # build initial vocabulary
    vocab_list: list[bytes] = [bytes([i]) for i in range(256)]
    for t in special_tokens:
        vocab_list.append(to_bytes(t))

    vocab = set(vocab_list)
    print_var(vocab)

    # iterate and find most common pair
    # for i in range(len(vocab), vocab_size + 1): # should use while loop instead
    for i in range(0, 10):
        total_pair_to_count = Counter()
        for wt in word_tuples_to_count:
            pair_to_count = get_vocab_pair_count_in_word(wt, vocab)
            for p in pair_to_count:
                total_pair_to_count[p] += word_tuples_to_count[wt] * pair_to_count[p]

        print_var(total_pair_to_count)

        most_frequent_pair = get_most_frequent_pair(total_pair_to_count)

        print_var(most_frequent_pair)
        vocab.add(most_frequent_pair)
        print("vocab size:", len(vocab))
        # need to change the word_tuples_to_count
        update_word_tuples_to_count(word_tuples_to_count, most_frequent_pair)

    return {}, []


def update_word_tuples_to_count(to_update: dict[tuple[bytes, ...], int], new_vocab: bytes):
    """updates the tuple[bytes, ...] and merge the common pairs."""
    # list(to_update) only copies a flat list of references (very lightweight). this is to avoid
    # changing a dict's length while iterating it, which python forbids.
    for word_tuple in list(to_update):
        for i in range(len(word_tuple) - 1):
            pair = word_tuple[i] + word_tuple[i + 1]
            if pair == new_vocab:
                new_word_tuple = get_new_word_tuple(word_tuple, new_vocab)
                # the new tuple shouldn't exist in the map.
                if new_word_tuple in to_update:
                    raise RuntimeError
                count = to_update[word_tuple]
                to_update[new_word_tuple] = count
                del to_update[word_tuple]
                break  # otherwise there is a crash since word_tuple is deleted from the map already.


def get_new_word_tuple(word_tuple: tuple[bytes, ...], new_vocab: bytes) -> tuple[bytes, ...]:
    """merges neighboring tuples if the pair is equal to new_vocab."""
    if len(word_tuple) < 2:
        return word_tuple

    new_word_list: list[bytes] = []
    for i in range(len(word_tuple) - 1):
        curr = word_tuple[i]
        next = word_tuple[i + 1]
        if curr + next == new_vocab:
            new_word_list.append(new_vocab)
        else:
            new_word_list.append(curr)

    return tuple(new_word_list)


def get_most_frequent_pair(pair_to_count: dict[bytes, int]) -> bytes:
    highest_count = 0
    most_common_pairs: list[bytes] = []
    for pair in pair_to_count:
        if pair_to_count[pair] > highest_count:
            highest_count = pair_to_count[pair]
            most_common_pairs = [pair]
        elif pair_to_count[pair] == highest_count:
            most_common_pairs.append(pair)

    most_common_pairs.sort(reverse=True)
    return most_common_pairs[0]


def to_bytes(w: str) -> bytes:
    return w.encode("utf-8")


def print_var(var):
    # 1. Look back at the frame of the code that called this function
    frame = inspect.currentframe().f_back

    # 2. Grab the exact line of text where print_var() was executed
    call_line = inspect.getframeinfo(frame).code_context[0].strip()

    # 3. Extract whatever text was placed inside the parentheses
    # e.g., "print_var(total_count)" -> "total_count"
    start = call_line.find("(") + 1
    end = call_line.rfind(")")
    var_name = call_line[start:end].strip()

    # 4. Print it out manually matching Python's f-string debug style
    print(f"{var_name}={repr(var)}")


def get_vocab_pair_count_in_word(word: tuple[bytes, ...], vocab: set[bytes]) -> dict[bytes, int]:
    pair_to_count = Counter()
    for i in range(len(word) - 1):
        pair = word[i] + word[i + 1]
        pair_to_count[pair] += 1

    return pair_to_count


if __name__ == "__main__":
    # get_vocab_pair_count_in_word((b'a', b'b', b'b', b'b', b'a'), {b'a', b'b'})
    run_train_bpe(sys.argv[1], 300, ["<|endoftext|>"])


###############################################################################################


def get_vocab_pair_count_in_word_deprecated(word: str, vocab: set[bytes]) -> dict[bytes, int]:
    """
    this is the minimal test example for the "find pair" algorithm.
    this was under the wrong assumption, so the algorithm is deprecated.
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
