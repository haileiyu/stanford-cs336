from typing import IO, Any, BinaryIO, Iterable, Iterator
from cs336_basics.run_train_bpe import get_split_pattern, PAT
import regex as re
from collections import Counter
from itertools import pairwise
import pickle


type bytes_pair = tuple[bytes, bytes]
type int_pair = tuple[int, int]


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges : list[tuple[bytes, bytes]], special_tokens:list[str] | None =None):
        self.vocab = vocab
        self.inverse_vocab : dict[bytes, int] = {}
        for i, v in self.vocab.items():
            self.inverse_vocab[v] = i

        # todo: remove self.vocab and self.merges
        self.merges = set[bytes_pair]()
        self.merges_int_pairs = set[int_pair]()
        for m in merges:
            self.merges.add(m)
            int_one = self.inverse_vocab[m[0]]
            int_two = self.inverse_vocab[m[1]]
            t : int_pair = (int_one, int_two)
            self.merges_int_pairs.add(t)

        if special_tokens:
            self.special_tokens : list[str] = special_tokens
        else:
            self.special_tokens : list[str] = []


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "wb") as f:  # note: binary mode
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(merges_filepath, "wb") as f:  # note: binary mode
            pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)
        return Tokenizer(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        # first, split the str into tuples of bytes
        word_tuples = self.pretokenize(text, self.special_tokens)

        # third, merge the tokens within words, and produce a list of ints
        res : list[int] = []
        for word_tuple in word_tuples:
            idxs = self.get_idx(word_tuple)
            res.extend(idxs)

        # finally, merge the int lists
        return res


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for e in self.encode(s):
                yield e


    # simply iterate the vocab and convert the ids back to the string
    def decode(self, ids: list[int]) -> str:
        s : str = ""
        for id in ids:
            s = s + self.vocab[id].decode("utf-8")
        return s


    def pretokenize(self, text: str, special_tokens: list[str]) -> list[tuple[bytes, ...]]:
        # first, split on the special tokens.
        if len(special_tokens) > 0:
            splitted = re.split(get_split_pattern(special_tokens), text)
        else:
            splitted = [text] # if no special token, do not split
        word_tuples : list[tuple[bytes, ...]] = []
        for p in splitted:
            if p in special_tokens:
                continue
            for m in re.finditer(PAT, p):
                word = m.group()
                bytes_tuple = tuple([bytes([x]) for x in word.encode("utf-8")])
                word_tuples.append(bytes_tuple)

        return word_tuples


    def get_idx(self, word_tuple: tuple[bytes, ...]) -> list[int]:
        word_tuple_idx = []
        for t in word_tuple:
            word_tuple_idx.append(self.inverse_vocab[t])

        input = word_tuple_idx
        output = self.merge(input)
        # keep merging until you cannot merge anymore
        while (len(output) < len(input)):
            input = output
            output = self.merge(input)

        return output

    
    def merge(self, word_tuple_idx : list[int]) -> list[int]:
        if len(word_tuple_idx) < 2:
            return word_tuple_idx

        new_word_tuple_idx = []
        i = 0
        found_merge = False
        for i in range(0, len(word_tuple_idx) - 1):
            p : int_pair = (word_tuple_idx[i], word_tuple_idx[i+1])
            if not p in self.merges_int_pairs:
                new_word_tuple_idx.append(word_tuple_idx[i])
            else:
                # maybe should build another index for this shit
                new_bytes = self.vocab[p[0]] + self.vocab[p[1]]
                new_word_tuple_idx.append(self.inverse_vocab[new_bytes])
                new_word_tuple_idx.extend(word_tuple_idx[i + 2:])
                found_merge = True
                # # now need to start over
                break # we only merge once per call

        # at the end
        if not found_merge:
            new_word_tuple_idx.append(word_tuple_idx[i + 1])

        return new_word_tuple_idx


if __name__ == "__main__":
    vocab : dict[int, bytes] = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges : list[bytes_pair] = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a',b't')]
    t = Tokenizer(vocab, merges, None)
    # text = "the cat ate"
    # print(t.encode(text))
    # input : tuple[bytes, ...] = (b' ', b'a', b't', b'e')
    # print(t.get_idx(input))
    # input : list[int] = [10, 3]
    # input : list[int] = [5, 4, 3]
    input : tuple[bytes, ...] = (b't', b'h', b'e')
    print(t.get_idx(input))



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