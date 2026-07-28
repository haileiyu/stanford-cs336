"""Tokenizer impl for cs336."""
# todo: parallelize the pretokenization process -- to use less memory overall, otherwise the process is killed.
# todo: in `encode`, there are various slices. investigate if those slicing would cause unnecessary copying.
from typing import Any, Iterable, Iterator
from cs336_basics.run_train_bpe import get_split_pattern, PAT
import regex as re
import pickle
from multiprocessing import Pool
import math


type bytes_pair = tuple[bytes, bytes]
type bytes_tuple = tuple[bytes, ...]
type int_pair = tuple[int, int]
NUM_PROCESSES = 8  # parallelization
CHUNK_SIZE = 100000


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.inverse_vocab: dict[bytes, int] = {}
        for i, v in vocab.items():
            self.inverse_vocab[v] = i

        self.idx_merges: dict[int_pair, int] = {}
        for i, m in enumerate(merges):
            t: int_pair = (self.inverse_vocab[m[0]], self.inverse_vocab[m[1]])
            self.idx_merges[t] = i

        if special_tokens:
            self.special_tokens: list[str] = special_tokens
            for s in special_tokens:
                # only give the special token an id if it doesn't already exist in the vocab
                encoded_bytes = s.encode("utf-8")
                if not encoded_bytes in self.inverse_vocab:
                    new_idx = len(self.vocab)
                    self.vocab[new_idx] = encoded_bytes
                    self.inverse_vocab[encoded_bytes] = new_idx

        else:
            self.special_tokens: list[str] = []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []
        with open(vocab_filepath, "rb") as f:  # note: binary mode
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:  # note: binary mode
            merges = pickle.load(f)
        return Tokenizer(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pretokenized = self._pretokenize(text, self.special_tokens)
        start = 0
        res = []
        while start < len(pretokenized):
            end = min(len(pretokenized), start + CHUNK_SIZE * NUM_PROCESSES)
            curr_working_set = pretokenized[start:end]
            # then split the working set into chunks
            num_processes = math.ceil(len(curr_working_set) / CHUNK_SIZE)
            input_list: list[list[bytes_tuple | str]] = []
            for i in range(0, num_processes):
                chunk_start = i * CHUNK_SIZE
                chunk_end = chunk_start + CHUNK_SIZE
                view = curr_working_set[chunk_start:chunk_end]
                input_list.append(view)
            with Pool(num_processes) as pool:
                results = pool.map(self._encode_chunk, input_list)
                for r in results:
                    res.extend(r)
            start = end
        return res

    def _encode_chunk(self, chunk: list[bytes_tuple | str]) -> list[int]:
        res: list[int] = []
        for p in chunk:
            if isinstance(p, str):
                # special token
                res.append(self.inverse_vocab[p.encode("utf-8")])
                continue
            idxs = self._convert_word_tuple_to_idx(p)
            res.extend(idxs)

        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for e in self.encode(s):
                yield e

    def decode(self, ids: list[int]) -> str:
        s: bytes = b""
        for id in ids:
            token = self.vocab[id]
            s = s + token
        return s.decode("utf-8", errors="replace")

    def _pretokenize(self, text: str, special_tokens: list[str]) -> list[bytes_tuple | str]:
        if len(special_tokens) > 0:
            # in get_split_pattern, we deliberately didn't use group so that the special tokens are discarded.
            # while that works for bpe, in the tokenizer we do need to keep the special tokens.
            split_pattern = "(" + get_split_pattern(special_tokens) + ")"
            splitted = re.split(split_pattern, text)
        else:
            splitted = [text]  # if no special token, do not split

        res = []
        for p in splitted:
            if p in special_tokens:
                res.append(p)
            else:
                for m in re.finditer(PAT, p):
                    word = m.group()
                    bytes_tuple = tuple([bytes([x]) for x in word.encode("utf-8")])
                    res.append(bytes_tuple)

        return res

    def _convert_word_tuple_to_idx(self, word_tuple: bytes_tuple) -> list[int]:
        word_tuple_idx = []
        for t in word_tuple:
            word_tuple_idx.append(self.inverse_vocab[t])

        input = word_tuple_idx
        output = self._merge(input)
        # keep merging until you cannot merge anymore
        while len(output) < len(input):
            input = output
            output = self._merge(input)

        return output

    def _merge(self, word_tuple_idx: list[int]) -> list[int]:
        if len(word_tuple_idx) < 2:
            return word_tuple_idx

        lowest_rank = len(self.idx_merges) + 1
        lowest_rank_idx = -1
        for i in range(0, len(word_tuple_idx) - 1):
            p: int_pair = (word_tuple_idx[i], word_tuple_idx[i + 1])
            if p in self.idx_merges:
                rank = self.idx_merges[p]
                if rank < lowest_rank:
                    lowest_rank = rank
                    lowest_rank_idx = i

        if lowest_rank_idx == -1:
            return word_tuple_idx

        new_bytes = self.vocab[word_tuple_idx[lowest_rank_idx]] + self.vocab[word_tuple_idx[lowest_rank_idx + 1]]
        new_idx = self.inverse_vocab[new_bytes]
        return word_tuple_idx[0:lowest_rank_idx] + [new_idx] + word_tuple_idx[lowest_rank_idx + 2 :]


if __name__ == "__main__":
    vocab: dict[int, bytes] = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }
    merges: list[bytes_pair] = [(b"t", b"h"), (b" ", b"c"), (b" ", b"a"), (b"th", b"e"), (b" a", b"t")]
    t = Tokenizer(vocab, merges, None)
    text = "the cat ate"
    print(t.encode(text))


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return Tokenizer(vocab, merges, special_tokens)
