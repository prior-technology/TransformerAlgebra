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
