# transformer accounting

page 27, problem (transformer_accounting).

## setup from the handout

> **Rule:** Given $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the matrix-matrix product $AB$ requires $2mnp$ FLOPs.
>
> To see this, note that $(AB)[i,j] = A[i,:] \cdot B[:,j]$, and that this dot product requires $n$ additions and $n$ multiplications ($2n$ FLOPs). Then, since the matrix-matrix product $AB$ has $m \times p$ entries, the total number of FLOPs is $(2n)(mp) = 2mnp$.

The vast majority of FLOPs in a Transformer are matrix multiplies, so the approach is:
write down all the matrix multiplies in a forward pass, then convert each one into FLOPs.

## question a

> Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:
>
> | parameter | value |
> | --- | --- |
> | `vocab_size` | 50,257 |
> | `context_length` | 1,024 |
> | `num_layers` | 48 |
> | `d_model` | 1,600 |
> | `num_heads` | 25 |
> | `d_ff` | 4,288 (the nearest multiple of 64 to $\frac{8}{3} \times 1{,}600$) |

> Suppose we constructed our model using this configuration. How many trainable parameters would our model have?

trainable parameters:
- embedding
- LM head
- ln final weight
- also some parameters that are per-block:
  - q, k, v, output_projection
  - w1, w2 and w3 in the feed forward layer
  - the weights in rmsnorm: ln1.weight and ln2.weight

> Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

single precision float is ... 32 bits, or 4 bytes.

to load the model, you need to load all the parameters.

- embedding: vocab_size * d_model
- LM head: vocab_size * d_model
- ln final weight: d_model
- also some parameters that are per-block:
  - q: d_model * d_model
  - k: same as q
  - v: same as q
  - output_projection: same as q
  - w1, w2 and w3 in the feed forward layer: 3 * d_model * d_ff
  - the weights in rmsnorm: ln1.weight and ln2.weight: d_model

so, the sum:

$$
\begin{aligned}
\text{params} &= 2 \cdot \text{vocab\_size} \cdot d_{\text{model}} + d_{\text{model}} + \text{num\_layers} \cdot \left(4 d_{\text{model}}^2 + 3 d_{\text{model}} d_{\text{ff}} + 2 d_{\text{model}}\right) \\
&= 2 \cdot 50257 \cdot 1600 + 1600 + 48 \cdot \left(4 \cdot 1600^2 + 3 \cdot 1600 \cdot 4288 + 2 \cdot 1600\right) \\
&= 1{,}640{,}452{,}800
\end{aligned}
$$

that's 6561811200 bytes, or roughly 6GB.

## question b

> Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.


*Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs required.*



--- lol why am i computing the flop, scratch, need update.

### flop calculation

from multihead_self_attention.py, flop is:
# = 8 * ... * seq * d_model^2
# + (4 * d_k + 5) * ... * num_heads * seq^2
# also, + 6 * ... * seq * d_model if rope is enabled

what does ... include?
- batch size, which is 1 i guess.

so: seq = 1024, num_heads = 25, d_model = 1600, and ... = 1
also, d_k is d_model / num_heads = 64

the transformer part's flop is:

8 * 1 * 1024 * 1600^2 + 261 * 1 * 25 * 1024^2 = 27813478400. per block. don't think there is rope.

flop = 48 * 27813478400 = 1.335e12


## question c

> Based on your analysis above, which parts of the model require the most FLOPs?


*Deliverable: A one-to-two sentence response.*

## question d

> Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads).


> As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?


*Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.*

## question e

> Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change?


> How does the relative contribution of FLOPs of the model components change?


*Deliverable: A one-to-two sentence response.*


