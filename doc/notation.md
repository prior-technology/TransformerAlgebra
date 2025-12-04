# Notation

Notation aims to be consistent with [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#notation) with some deviations.

## Indexing Conventions

- **Superscripts**: block number (0 = embedding layer, 1 to $L$ = transformer blocks)
- **Subscripts**: token position (0 = first token, $n-1$ = last token)

## Core Symbols

| Symbol | Meaning |
|--------|---------|
| $x^i_j$ | Residual vector after block $i$, at position $j$ |
| $\underline{\text{token}}$ | Token embedding vector (row of $W_E$) |
| $\overline{\text{token}}$ | Unembedding vector (row of $W_U$) |
| $\Delta x^i_j$ | Contribution from block $i$ to residual at position $j$ |
| $W_E$, $W_U$ | Embedding and unembedding matrices |
| $p_j$ | Positional embedding at position $j$ |

## Operators

| Symbol | Meaning |
|--------|---------|
| $LN$ | Layer normalization |
| $A^{i,h}$ | Attention pattern matrix for layer $i$, head $h$ |
| $\tilde{A}^i$ | Attention sub-layer with pre-norm: $\tilde{A}^i(x) = \text{Attn}^i(LN(x))$ |
| $M^i$ | MLP layer $i$ |
| $\tilde{M}^i$ | MLP sub-layer with pre-norm: $\tilde{M}^i(x) = M^i(LN(x))$ |
| $B_i$ | Full transformer block $i$ |

The tilde notation $\tilde{A}$, $\tilde{M}$ indicates sub-layers with their preceding layer normalization absorbed.
