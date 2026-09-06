# batch size experiment

kept the token budget constant -- when batch_size increased, decreased the number of training steps accordingly.

## round 1: fixed max learning rate

let's fix the max learning rate at a heuristic value (1e-3) and compare different batch sizes.

batch size: 16, 32, 64.

### graphs

![fixed lr lr](<batch_size experiment fixed lr lr.png>)

![fixed lr loss](<batch_size experiment fixed lr loss.png>)

### observation

#### learning rate

the learning rate dropped faster when batch size is larger. this is expected, because the larger batch_size, the less overall training steps, and the
learning rate has to drop faster to keep up with training steps.

#### loss

looking at the last single step, the losses look about the same. but the last step bounces around a lot, so the
fair comparison is the average loss over the last 10% of steps. by that measure the loss gets worse as batch
size grows: 1.6346 (16), 1.6547 (32), 1.6922 (64). the gaps are much larger than the +-0.003 error bars, so the
ordering is real.

| batch size | max lr | steps | mean over last 10% of steps | sem   | sd of one logged point | last single step |
| ---------- | ------ | ----- | --------------------------- | ----- | ---------------------- | ---------------- |
| 16         | 1.0e-3 | 9765  | 1.6346                      | 0.003 | 0.087                  | 1.6829           |
| 32         | 1.0e-3 | 4882  | 1.6547                      | 0.003 | 0.060                  | 1.5458           |
| 64         | 1.0e-3 | 2441  | 1.6922                      | 0.003 | 0.046                  | 1.7208           |

according to claude, it's expected because the training steps decreased, and the number of optimizations decreased too.

loss band width: got narrower as the batch size increased. this is because the average loss is averaged over batch_size. it doesn't mean the actuall loss is lower, it's just that the seen loss is less noisy.

## round 2: variable max learning rate

did a sweeping of batch size from 16, 32, 64, 128. the learning rate was also tuned according to the batch_size: `lr = 1e-3 * (batch_size / 32) ** 0.5` (this isn't documented in the course assignment, but is, per claude, a common heuristic).

### graphs

the learning rate graph is:
![var lr lr](<batch_size_experiment_var_lr_lr.png>)

![var lr loss](<batch_size_experiment_var_lr_loss.png>)

### observations

#### loss

the loss can be bouncey so we shouldn't look at the final loss in wandb. instead, should get the average of last 10% steps.

| batch size | max lr | steps | mean over last 10% of steps | sem   | sd of one logged point | last single step |
| ---------- | ------ | ----- | --------------------------- | ----- | ---------------------- | ---------------- |
| 16         | 7.1e-4 | 9765  | 1.6418                      | 0.003 | 0.089                  | 1.7387           |
| 32         | 1.0e-3 | 4882  | 1.6491                      | 0.003 | 0.060                  | 1.6015           |
| 64         | 1.4e-3 | 2441  | 1.6672                      | 0.003 | 0.042                  | 1.6338           |
| 128        | 2.0e-3 | 1220  | 1.7051                      | 0.003 | 0.032                  | 1.7066           |

as the batch size increased (and max lr increased too), the loss increased (same as round 1), but the loss increased slower with larger learning rate.

why it increased slower with larger lr? larger step helped mitigating the downsides of fewer optimizations.

#### memory usage

the Process Memory Available (MB) chart showed that as batch_size increased, the memory decreased. it's not clear to me whether that chart is "memory in use" or "memory not being used". if former, why smaller size uses more memory? i'm expecting the opposite. if the latter, why do we care about this at all?

claude says it's the free memory. still not sure why we care about it. but one possibility -- when you train on mps, it will eat the unified memory, but that usage isn't reflected on the python program (which is on cpu). the free memory might be a better indicator of memory usage.

#### loss band width

same as round 1, the width decreased due to larger batch_size. but it's a measurement effect.
