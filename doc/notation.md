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

## Layer Normalization

Different layer norms are distinguished by their position in the architecture:

| Symbol | Meaning |
|--------|---------|
| $LN_A^i$ | Pre-attention layer norm at block $i$ |
| $LN_M^i$ | Pre-MLP layer norm at block $i$ |
| $LN^T$ | Terminal (final) layer norm before unembedding |

Note: $LN_A^i$ normalizes the input before it splits into attention heads—all heads at block $i$ share the same normalized input.

## Operators

| Symbol | Meaning |
|--------|---------|
| $A^{i,h}$ | Attention pattern matrix for layer $i$, head $h$ |
| $\tilde{A}^i$ | Attention sub-layer with pre-norm: $\tilde{A}^i(x) = \text{Attn}^i(LN_A^i(x))$ |
| $M^i$ | MLP layer $i$ |
| $\tilde{M}^i$ | MLP sub-layer with pre-norm: $\tilde{M}^i(x) = M^i(LN_M^i(x))$ |
| $B_i$ | Full transformer block $i$ |

The tilde notation $\tilde{A}$, $\tilde{M}$ indicates sub-layers with their preceding layer normalization absorbed.

## Prompted Transformers

| Expression | Meaning |
|------------|----------|
| $T$ | A transformer (weights only, no context) |
| $T(c)$ | Prompted transformer with context $c$ |
| $T(t_1, \ldots, t_n)$ | Transformer with tokens $t_1, \ldots, t_n$ in context |
| $T(\text{"Some context"})$ | Transformer with literal text as context |
| $T(c)(x)$ | Action of prompted transformer on embedding $x$ at next position |
| $\mathcal{C}(c)$ | Context state: all residuals $\{x^i_j\}$ for context $c$ |

Context concatenation: $T(c_1, c_2)$ is equivalent to $T(c_1 \cdot c_2)$ where $\cdot$ denotes token sequence concatenation.

## Truncated Transformer

| Expression | Meaning |
|------------|----------|
| $T^i$ | Transformer truncated after block $i$ (first $i$ blocks only) |
| $T^0(x)$ | Identity: $T^0(x) = x$ (embedding, before any blocks) |
| $T^n(x)$ | Residual after all $n$ blocks (before final layer norm) |

**Important**: $T^i(x)$ produces the *unnormalized* residual. The full transformer applies the terminal layer norm:
$$T(x) = LN^T(T^n(x))$$

The block contribution $\Delta x^i$ is defined as the difference:
$$\Delta x^i = T^{i+1}(x) - T^i(x)$$

## Expansion Identity

The unnormalized residual $T^n(x)$ can be expanded as a sum of contributions:

**Compact form:**
$$T^n(x) = x + \sum_{i=0}^{n-1} \Delta x^i$$

**Expanded form:**
$$T^n(x) = x + \Delta x^0 + \Delta x^1 + \cdots + \Delta x^{n-1}$$

The full transformer output applies the terminal layer norm:
$$T(x) = LN^T(T^n(x)) = LN^T\left(x + \sum_{i=0}^{n-1} \Delta x^i\right)$$

Note: Each $\Delta x^i$ depends on the accumulated residual $T^i(x)$, not just the original embedding. For parallel attention architectures (e.g., Pythia/GPT-NeoX):
$$\Delta x^i = \tilde{A}^i(T^i(x)) + \tilde{M}^i(T^i(x))$$
