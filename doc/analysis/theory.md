# Transformer Theory

This document covers well-established theory for decoder-only transformer models, using Pythia as a reference implementation.

## References

- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- [Neel Nanda's Mechanistic Interpretability Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary)

## Inference and Logits

The unembedding weights are applied to the final layer residual vectors to generate logits (unbounded scores normalized via softmax to probabilities). The predicted token is selected from those with the greatest logits.

For output residual vector $x_j^L$ at position $j$, the logits are:
$$
W_U \cdot LN(x_j^L)
$$

The logit for a particular token is the dot product:
$$
\langle \overline{\text{token}}, LN(x_j^L) \rangle
$$

### From Logits to Probabilities

During inference, the probability of selecting a particular next token is computed via softmax over the full vocabulary:
$$
P(\text{token} | x_j^L) = \frac{\exp(\langle \overline{\text{token}}, LN(x_j^L) \rangle)}{\sum_{t \in V} \exp(\langle \overline{t}, LN(x_j^L) \rangle)}
$$

This normalizes the logits into a probability distribution. The denominator sums over all tokens $t$ in the vocabulary $V$, making the probabilities sum to 1. This is the fundamental connection between the geometric view (inner products) and the probabilistic view (next-token prediction).

## Residual Stream Structure

The layers of the transformer add to the residual stream:
$$
x_j^L = x_j^0 + \sum_{i=1}^{L} \Delta x_j^i
$$

where $x_j^0 = \underline{\text{token}_j} + p_j$ combines the token embedding with positional embedding.

## Block Structure (Pre-Norm)

For Pythia and similar pre-norm architectures, each block applies layer normalization *before* each sub-layer. The **block operator**:

$$
B_i(x) = x + \tilde{A}^i(x) + \tilde{M}^i(x + \tilde{A}^i(x))
$$

where:
- $\tilde{A}^i(x) = \text{Attn}^i(LN(x))$ — attention with pre-norm
- $\tilde{M}^i(y) = M^i(LN(y))$ — MLP with pre-norm

The full forward pass is:
$$
x_j^L = B_L \circ B_{L-1} \circ \cdots \circ B_1 (x_j^0)
$$

with a final layer norm before unembedding: $\text{logits}_j = W_U \cdot LN(x_j^L)$

## Block Contribution

The contribution from block $i$ can be decomposed:
$$
\Delta x_j^i = \tilde{A}^i(x_j^{i-1}) + \tilde{M}^i(x_j^{i-1} + \tilde{A}^i(x_j^{i-1}))
$$

Note: The MLP sees the residual *after* attention has been added, so these terms are not independent.

## Information Flow and Causal Structure

Information flow in a decoder-only transformer is constrained by two factors:

1. **Causal masking**: Position $j$ cannot attend to positions $k > j$
2. **Sequential computation**: Block $i$ computes before block $i+1$

This creates a "light cone" constraint. The contribution $\Delta x_j^i$ can only depend on:
- Residuals $\{x_k^{i-1}\}_{k \leq j}$ (current and earlier positions, one layer below)
- The weights of block $i$

More precisely:
$$\Delta x_j^i = f\left(\{x_k^{i-1}\}_{k \leq j}, \theta_i\right)$$

where $\theta_i$ denotes the parameters of block $i$.

```
Layer:  3    ·  ·  ·  ●──────┐
        2    ·  ·  ●  ↑      │ Causal
        1    ·  ●  ↑  ↑      │ light cone
        0    ●──●──●──●──────┘
Position:    0  1  2  3
```

**Unlike physical light cones, there is no "speed of information" limit.** Attention allows direct transfer across arbitrary position distances in a single layer. Information from $(j=0, i=0)$ can influence $(j=3, i=1)$ directly—the attention at layer 1 can read from all earlier positions at layer 0.

This is a key advantage over recurrent architectures, where information must traverse each position sequentially, creating a bottleneck for long-range dependencies.

### References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — introduces the transformer and attention mechanism
- Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — formalizes information flow in terms of residual stream contributions

## Non-Linear Operator Properties

| Component | Property |
|-----------|----------|
| **Attention** | Data-dependent linear map—the pattern $A^{i,h}$ depends on the input, making the overall operation non-linear |
| **MLP** | Position-wise non-linear transformation (GeLU activation in Pythia) |
| **Layer Norm** | Non-linear scaling that projects onto a sphere (after centering) |

Because each $B_i$ has the structure $x + f_i(x)$ (adding a perturbation rather than fully transforming), contributions from different layers can be meaningfully compared in the shared residual space.

## Layer Normalization and Dot Product

$$\langle v_1, LN(v_2) \rangle \approx |v_1| \cos{\theta_{v_1,c(v_2)}} $$

where $\theta_{a,b}$ is the angle between vectors $a$ and $b$, and $c(v)$ is the centering operation.

Each vector sum can be considered in 2D, where it either increases or decreases $\theta$. By building up the sum of vectors on the right we can see which vectors contribute to the angle used in the final dot product.

## Prompted Transformers

A **prompted transformer** $T(c)$ represents a transformer with context $c$ already processed. It acts on the next token's embedding to produce a prediction.

### Context State

Let $\mathcal{C}(c) = \{x^i_j\}$ be the **context state**—all residual vectors at all positions and layers for context $c$. The prompted transformer's action on embedding $x$ is:

$$T(c)(x) = F(x, \mathcal{C}(c))$$

where $F$ is the forward pass for the final position, which can attend to all of $\mathcal{C}$.

### Causal Structure

For decoder-only transformers with causal masking, earlier positions are computed identically regardless of later tokens:

$$x^i_j \text{ in } T(t_1, \ldots, t_n) = x^i_j \text{ in } T(t_1, \ldots, t_n, t_{n+1}) \quad \text{for } j < n$$

This means context states *extend* rather than change:

$$\mathcal{C}(c_1, c_2) \supset \mathcal{C}(c_1)$$

The residuals for $c_1$ are preserved; $c_2$ adds new positions that can attend to them.

### Context Does Not Compose Additively

Importantly:
$$T(c_1, c_2)(x) \neq T(c_1)(x) + T(c_2)(x)$$

Context is not additive. The relationship is:
- $T(c_1)$ establishes residual states
- $T(c_1, c_2)$ extends those states; new positions attend to old ones
- The final action on $x$ depends on the *full* context state
