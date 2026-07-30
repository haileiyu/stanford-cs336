import sys
import time
import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_doc(doc_path: str, vocab_path: str, merges_path: str, output_path: str | None) -> list[int]:
    """Encodes doc, and writes the integer into a file."""
    vocab : dict[int, bytes] = {}
    merges : list[tuple[bytes, bytes]] = []
    t = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    doc = open(doc_path).read()
    ids = t.encode(doc)
    if output_path:
        arr = np.array(ids, dtype=np.uint16)
        np.save(output_path, arr)
        # with open(output_path, "wb") as f:  # note: binary mode
        #     pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ids


def calculate_compression_ratio(doc_path: str, vocab_path: str, merges_path: str):
    """compression ratio is defined as bytes/tokens"""
    with open(doc_path, "rb") as f:
        doc_bytes = len(f.read())
        ids = encode_doc(doc_path, vocab_path, merges_path, None)
        ratio = doc_bytes / len(ids)
        print(ratio)


def estimate_throughput(doc_path: str, vocab_path: str, merges_path: str):
    """let's train on a 2gb data set, and time it."""
    start = time.perf_counter()
    ids = encode_doc(doc_path, vocab_path, merges_path, None)
    elapsed = time.perf_counter() - start
    print(f"took {elapsed:.3f} s")
    
    return 0



if __name__ == "__main__":
    # sample 10 documents from datasets.
    # docs = open(sys.argv[1]).read().split("<|endoftext|>")
    # sample = random.sample([d for d in docs if d.strip()], 10)
    # open(sys.argv[2],"w").write("\n<|endoftext|>\n".join(sample))
    # calculate_compression_ratio(sys.argv[1], sys.argv[2], sys.argv[3])
    encode_doc(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    
