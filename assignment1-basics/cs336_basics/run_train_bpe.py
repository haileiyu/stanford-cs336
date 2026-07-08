import os
import sys
import regex as re
import inspect
from collections import Counter


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
type bytes_pair = tuple[bytes, bytes]


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
    # pretokenization
    word_tuple_to_count = get_word_tuple_to_count(input_path, special_tokens)

    for w in word_tuple_to_count:
        for b in w:
            if len(b) != 1:
                print(f"word: {w}, vocab: {b}")

    # build initial vocabulary
    vocab_list: list[bytes] = [bytes([i]) for i in range(256)]
    for t in special_tokens:
        vocab_list.append(to_bytes(t))

    vocab = {b: i for i, b in enumerate(vocab_list)}

    # iterate and find most common pair
    merges: list[bytes_pair] = []
    while len(vocab) < vocab_size:
        total_pair_to_count : Counter[bytes_pair] = Counter()
        for word_tuple in word_tuple_to_count:
            # update the dict directly
            update_total_pair_to_count(total_pair_to_count, word_tuple, word_tuple_to_count[word_tuple])

        most_frequent_pair = get_most_frequent_pair(total_pair_to_count)
        merges.append(most_frequent_pair)

        new_vocab = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[new_vocab] = len(vocab)
        update_word_tuples_to_count(word_tuple_to_count, new_vocab)
    
    return {value: key for key, value in vocab.items()}, merges


def update_total_pair_to_count(total_pair_to_count: Counter[bytes_pair], word_tuple: tuple[bytes, ...], word_tuple_count: int):
    for i in range(len(word_tuple) - 1):
        pair = (word_tuple[i], word_tuple[i + 1])
        total_pair_to_count[pair] += word_tuple_count


def get_word_tuple_to_count(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "r") as content_file:
        content = content_file.read()

    # first, split on the special tokens.
    escaped_special_tokens = []
    for special_token in special_tokens:
        escaped = re.escape(special_token)
        escaped_special_tokens.append(escaped)

    # make sure not to use group so that we skip the special tokens themselves.
    split_pattern = "|".join(escaped_special_tokens)

    splitted = re.split(split_pattern, content)
    word_to_count = Counter()
    for p in splitted:
        if p in special_tokens:
            continue
        for m in re.finditer(PAT, p):
            word_to_count[m.group()] += 1

    word_tuples_to_count: dict[tuple[bytes, ...], int] = {}
    for word in word_to_count:
        # split the word into tuples
        bytes_tuple = tuple([bytes([x]) for x in word.encode("utf-8")])
        word_tuples_to_count[bytes_tuple] = word_to_count[word]

    return word_tuples_to_count


def update_word_tuples_to_count(to_update: dict[tuple[bytes, ...], int], new_vocab: bytes):
    """updates the tuple[bytes, ...] and merge the common pairs."""
    # list(to_update) only copies a flat list of references (very lightweight). this is to avoid
    # changing a dict's length while iterating it, which python forbids.
    for word_tuple in list(to_update):
        for i in range(len(word_tuple) - 1):
            pair = word_tuple[i] + word_tuple[i + 1]
            if pair == new_vocab:
                new_word_tuple = get_new_word_tuple(word_tuple, new_vocab)
                # the new tuple shouldn't exist in the map, because a new word tuple contains
                # something we've never seen before.
                # well is it possible that the new word tuple was added to the dict earlier in
                # this forloop? yes, e.g. if the new_vocab is abc, it could come from ab + c, and a + bc.
                count = to_update[word_tuple]
                to_update[new_word_tuple] = to_update.get(new_word_tuple, 0) + count
                del to_update[word_tuple]
                i += 1  # todo: does this line make sense at all? i was fixing something in get_new_word_tuple fwiw.
                break  # otherwise there is a crash since word_tuple is deleted from the map already.


def get_new_word_tuple(word_tuple: tuple[bytes, ...], new_vocab: bytes) -> tuple[bytes, ...]:
    """merges neighboring tuples if the pair is equal to new_vocab."""
    if len(word_tuple) < 2:
        return word_tuple

    new_word_list: list[bytes] = []
    found_match_in_last_iter = False
    for i in range(len(word_tuple)):
        # if we found a match in the previous iter, we should simply skip this tuple element, since
        # it's now merged with previous
        if found_match_in_last_iter:
            found_match_in_last_iter = False
            continue

        curr = word_tuple[i]
        # if this is the last element, we simply add it
        if i == len(word_tuple) - 1:
            new_word_list.append(curr)
            break

        next = word_tuple[i + 1]
        if curr + next == new_vocab:
            found_match_in_last_iter = True
            new_word_list.append(new_vocab)
        else:
            new_word_list.append(curr)

    return tuple(new_word_list)


def get_most_frequent_pair(pair_to_count: dict[bytes_pair, int]) -> bytes_pair:
    highest_count = 0
    most_common_pair_ties: list[bytes_pair] = []
    for pair in pair_to_count:
        if pair_to_count[pair] > highest_count:
            highest_count = pair_to_count[pair]
            most_common_pair_ties = [pair]
        elif pair_to_count[pair] == highest_count:
            most_common_pair_ties.append(pair)

    if len(most_common_pair_ties) == 0:
        raise RuntimeError
    return max(most_common_pair_ties)


def to_bytes(w: str) -> bytes:
    return w.encode("utf-8")


def print_var(var):
    """this helper function is created by gemini."""
    # 1. Look back at the frame of the code that called this function
    frame = inspect.currentframe().f_back  # type: ignore

    # 2. Grab the exact line of text where print_var() was executed
    call_line = inspect.getframeinfo(frame).code_context[0].strip()  # type: ignore

    # 3. Extract whatever text was placed inside the parentheses
    # e.g., "print_var(total_count)" -> "total_count"
    start = call_line.find("(") + 1
    end = call_line.rfind(")")
    var_name = call_line[start:end].strip()

    # 4. Print it out manually matching Python's f-string debug style
    print(f"{var_name}={repr(var)}")


if __name__ == "__main__":
    run_train_bpe(sys.argv[1], 500, ["<|endoftext|>"])
