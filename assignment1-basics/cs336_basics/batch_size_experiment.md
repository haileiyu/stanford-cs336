# batch size experiment

did a sweeping of batch size from 16, 32, 64, 128. to keep the token budget constant, i've adjusted the num_iterations accordingly (inverse proportional to batch size). the learning rate was also tuned according to the batch_size: `lr = 1e-3 * (batch_size / 32) ** 0.5` (this isn't documented in the course assignment, but is, per claude, the industrial standard).

the graph is as follows:

![batch_size experiment](<batch_size_experiment.png>)

final loss:
- 16: 1.7387382984161377
- 32: 1.6014947891235352
- 64: 1.6337835788726809
- 128: 1.7066335678100586

## observations

### loss

the loss was smallest at 32. not sure why.

claude noted that i shouldn't look at the final loss. instead, should look at the final 10%'s average:

| batch size | max lr | steps | last single step | mean over last 10% of tokens | sd of one logged point |
|---|---|---|---|---|---|
| 16  | 7.1e-4 | 9765 | 1.7387 | 1.6418 ± 0.003 | 0.089 |
| 32  | 1.0e-3 | 4882 | 1.6015 | 1.6491 ± 0.003 | 0.060 |
| 64  | 1.4e-3 | 2441 | 1.6338 | 1.6672 ± 0.003 | 0.042 |
| 128 | 2.0e-3 | 1220 | 1.7066 | 1.7051 ± 0.003 | 0.032 |

claude explains that it's because we pushed the batch size beyond the critical batch size. the num_iterations decreased, so the optimizer didn't have a lot of chance to reduce the loss. so the curve would be flat then increases. we should stay in the flat area (32) to keep the hardware utilized without costing loss function.

### memory usage

the Process Memory Available (MB) chart showed that as batch_size increased, the memory decreased. it's not clear to me whether that chart is "memory in use" or "memory not being used". if former, why smaller size uses more memory? i'm expecting the opposite. if the latter, why do we care about this at all?

claude says it's the free memory. still not sure why we care about it. but one possibility -- when you train on mps, it will eat the unified memory, but that usage isn't reflected on the python program (which is on cpu). the free memory might be a better indicator of memory usage.

### loss band width

another observation is that the green band (smaller batch_size) is wider, meaning the loss function had a bigger oscillation during training. that doesn't make sense, since the learning rate was smaller, and i expected a smaller loss.

claude told me it's because, when batch_size is smaller, because the loss function is computed on a random handful of examples, and the smaller the batch is, and the larger variance the loss function is. when batch_size increases, it's more steady.

# next
- do another round of experiment, where the learning rate doesn't change with batch_size, and see what happens.

## round 2: keep learning rate fixed, try 16, 32, and 64.

