import regex as re

# text = """
# Everyone was happy it was all back to ordinary.
# <|endoftext|>
# Once upon a time, there was a little girl named Mia.
# """

# text = "dog.<|endoftext|>Once"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# special_token = "<|endoftext|>"
# escaped_special_token = re.escape(special_token)

# special_tokens = ["<|endoftext|>", "dog"]

# splitted = re.split('(' + escaped_special_token + ')', text)
# for p in splitted:
#   if p == special_token:
#     print(p)
#     continue
#   for m in re.finditer(PAT, p):
#     print(m.group())

# now let's try multiple tokens
from collections import Counter

content = """
Everyone was happy it was all back to ordinary.
<|endoftext|>
Once upon a time, there was a little girl named Mia.
"""

special_tokens = ["<|endoftext|>", "all"]

escaped_special_tokens = []
for special_token in special_tokens:
    escaped = re.escape(special_token)
    escaped_special_tokens.append(escaped)

split_pattern = "|".join(escaped_special_tokens)

splitted = re.split("(" + split_pattern + ")", content)
word_to_count = Counter()
for p in splitted:
    if p in special_tokens:
        print("found special token:", p)
        continue
    for m in re.finditer(PAT, p):
        print(m.group())
        word_to_count[m.group()] += 1

print(word_to_count)
