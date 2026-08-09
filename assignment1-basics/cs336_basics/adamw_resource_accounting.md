
Transformer block
- RMSNorm(s)
- Multi-head self-attention sublayer: 𝑄𝐾𝑉 projections, 𝑄𝐾⊤ matrix multiply, softmax,
weighted sum of values, output projection.
- Position-wise feed-forward (SwiGLU): 𝑊 1, 𝑊 2, SiLU on the gate branch, element-wise
product, 𝑊 3
32
- final RMSNorm
- output embedding
- cross-entropy on logits


parameters:
- batch_size
- model hyperparameters (vocab_size, context_length, num_layers, d_model, num_heads). Assume d_ff = 8/3 × d_model.
- (note to self) d_k is d_model // num_heads

# scratch area

## intermediate params:


### memory usage

RMSNorm: 2 * d_model.
Multi-head self-attention:
- init:
  - rope:
    - init: max_seq_len * d_k // 2 * 3
    - forward: batch * context_length, d_k // 2 + batch * context_length * d_k // 2 = batch * context_length * d_model // num_heads
- 