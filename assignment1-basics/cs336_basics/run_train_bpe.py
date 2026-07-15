import os
import sys
import regex as re
import inspect
from collections import Counter
from typing import BinaryIO
from multiprocessing import Pool, cpu_count
from line_profiler import profile
from itertools import pairwise




PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CHUNK_SIZE = 100000000
type bytes_pair = tuple[bytes, bytes]


@profile
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
    word_tuple_to_count = load_word_tuple_to_count(input_path, special_tokens)

    # build initial vocabulary
    vocab_list: list[bytes] = [bytes([i]) for i in range(256)]
    for t in special_tokens:
        vocab_list.append(t.encode("utf-8"))

    vocab = {b: i for i, b in enumerate(vocab_list)}

    # iterate and find most common pair
    merges: list[bytes_pair] = []
    # build initial total_pair_to_count
    pair_to_count, pair_to_word_tuples = get_pair_to_count(word_tuple_to_count)

    while len(vocab) < vocab_size:
        # todo: should store the most frequent pairs, and the words that contain it
        most_frequent_pair = get_most_frequent_pair(pair_to_count)
        merges.append(most_frequent_pair)

        new_vocab = most_frequent_pair[0] + most_frequent_pair[1]
        vocab[new_vocab] = len(vocab)
        update_counts(word_tuple_to_count, pair_to_count, pair_to_word_tuples, most_frequent_pair)
    
    return {value: key for key, value in vocab.items()}, merges


def load_word_tuple_to_count(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    word_tuple_to_count: Counter[tuple[bytes, ...]] = Counter()
    # read the file in chunks
    with open(input_path, "rb") as f:
        # if we split on all the special characters, we will over-decompose and add a million tasks to the pool
        filesize = os.fstat(f.fileno()).st_size
        num_chunks = int(filesize / CHUNK_SIZE) + 1
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        print(boundaries)

        results = []
        with Pool(processes=cpu_count()) as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8")
                result = pool.apply_async(get_word_tuple_to_count_from_chunk, (chunk, special_tokens, ))
                results.append(result)

            print('process count: ', len(results))
            for r in results:
                wt_to_count = r.get()
                word_tuple_to_count.update(wt_to_count)

    return word_tuple_to_count


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Note: this is copied from the pretokenization_example.py.
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_pair_to_count(word_tuple_to_count: dict[tuple[bytes, ...], int]):
    pair_to_count : Counter[bytes_pair] = Counter()
    pair_to_word_tuples : dict[bytes_pair, set[tuple[bytes, ...]]] = {}

    for word_tuple, word_tuple_count in word_tuple_to_count.items():
        for pair in pairwise(word_tuple):
            pair_to_count[pair] += word_tuple_count
            if pair not in pair_to_word_tuples:
                pair_to_word_tuples[pair] = set()
            pair_to_word_tuples[pair].add(word_tuple)

    return pair_to_count, pair_to_word_tuples


def get_word_tuple_to_count_from_chunk(content: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    # first, split on the special tokens.
    splitted = re.split(get_split_pattern(special_tokens), content)
    word_to_count = Counter()
    for p in splitted:
        if p in special_tokens:
            continue
        for m in re.finditer(PAT, p):
            word_to_count[m.group()] += 1

    word_tuples_to_count: Counter[tuple[bytes, ...]] = Counter()
    for word in word_to_count:
        # split the word into tuples
        bytes_tuple = tuple([bytes([x]) for x in word.encode("utf-8")])
        word_tuples_to_count[bytes_tuple] = word_to_count[word]

    return word_tuples_to_count


def get_split_pattern(special_tokens: list[str]) -> str:
    escaped_special_tokens = []
    for special_token in special_tokens:
        escaped = re.escape(special_token)
        escaped_special_tokens.append(escaped)

    # make sure not to use group so that we skip the special tokens themselves.
    return "|".join(escaped_special_tokens)


@profile
def update_counts(word_tuple_to_count: dict[tuple[bytes, ...], int], pair_to_count: Counter[bytes_pair], pair_to_word_tuples: dict[bytes_pair, set[tuple[bytes, ...]]], most_frequent_pair: bytes_pair):
    """updates the tuple[bytes, ...] and merge the common pairs."""
    # list(word_tuples_to_count) only copies a flat list of references (very lightweight). this is to avoid
    # changing a dict's length while iterating it, which python forbids.
    # for word_tuple in list(word_tuple_to_count):
    # should likely remove this entry after
    tuples = pair_to_word_tuples[most_frequent_pair]
    for word_tuple in tuples:
    # for word_tuple in pair_to_word_tuples[new_vocab_pair]:
        # match_count = count_new_vocab_matches(word_tuple, most_frequent_pair)
        # todo: get rid of this check, because match count is certainly positive. maybe add assert instead.
        # if match_count > 0:
        new_word_tuple = get_new_word_tuple(word_tuple, most_frequent_pair)

        # todo: remove the old contribution
        for pair in pairwise(new_word_tuple):
            if pair not in pair_to_word_tuples:
                pair_to_word_tuples[pair] = set()
            pair_to_word_tuples[pair].add(new_word_tuple)

        # update word_tuple_to_count
        word_count = word_tuple_to_count[word_tuple]
        word_tuple_to_count[new_word_tuple] = word_tuple_to_count.get(new_word_tuple, 0) + word_count
        del word_tuple_to_count[word_tuple]

        # update total_pair_to_count: remove all contributions of old word tuple, then add new
        to_remove = get_pair_to_count_for_tuple(word_tuple)
        for pair, pair_count in to_remove.items():
            pair_to_count[pair] -= word_count * pair_count

        to_add = get_pair_to_count_for_tuple(new_word_tuple)
        for pair, pair_count in to_add.items():
            pair_to_count[pair] += word_count * pair_count


def get_pair_to_count_for_tuple(word_tuple: tuple[bytes, ...]) -> Counter[bytes_pair]:
    pair_to_count : Counter[bytes_pair] = Counter()
    for pair in pairwise(word_tuple):
        pair_to_count[pair] += 1

    return pair_to_count


def count_new_vocab_matches(word_tuple:tuple[bytes, ...], new_vocab_pair: bytes_pair) -> int:
    matches = 0
    for pair in pairwise(word_tuple):
        if pair == new_vocab_pair:
            matches += 1
    return matches


def get_new_word_tuple(word_tuple: tuple[bytes, ...], new_vocab_pair: bytes_pair) -> tuple[bytes, ...]:
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
        if (curr, next) == new_vocab_pair:
            found_match_in_last_iter = True
            new_word_list.append(curr + next)
        else:
            new_word_list.append(curr)

    return tuple(new_word_list)


def get_most_frequent_pair(pair_to_count: dict[bytes_pair, int]) -> bytes_pair:
    highest_count = 0
    most_common_pairs: list[bytes_pair] = []
    for pair in pair_to_count:
        if pair_to_count[pair] > highest_count:
            highest_count = pair_to_count[pair]
            most_common_pairs = [pair]
        elif pair_to_count[pair] == highest_count:
            most_common_pairs.append(pair)

    if len(most_common_pairs) == 0:
        raise RuntimeError
    return max(most_common_pairs)


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
    run_train_bpe(sys.argv[1], int(sys.argv[2]), ["<|endoftext|>"])