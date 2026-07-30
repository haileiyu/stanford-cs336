# my answers

this doc contains my answers to the non-coding questions.

## tokenizer_experiment (2.7)

a: sample 10 document and calculate the compression ratio (byte/token).

tinystories: 4.18
owt: 4.60

b: what if you tokenize owt with tiny stories?

3.341278502096376. significantly lower than tinystories-on-tinystories, or owt-on-owt.

c: throughput estimate.

took 1507.347 seconds to tokenize the 2.3GB data set. the throughput is 1.56 MB/s.



d: 