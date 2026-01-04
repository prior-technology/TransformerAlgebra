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

### Layer Norm Parameters

Each layer norm has learned parameters $\gamma$ (scale) and $\beta$ (shift), plus a data-dependent scale factor:

| Symbol | Meaning |
|--------|---------|
| $\gamma^i_A$, $\gamma^i_M$, $\gamma^T$ | Scale parameters for each layer norm |
| $\beta^i_A$, $\beta^i_M$, $\beta^T$ | Shift parameters for each layer norm |
| $\sigma(x)$ | Standard deviation: $\sigma(x) = \|P(x)\|/\sqrt{N}$ where $P$ is mean-centering |

The full layer norm operation:
$$LN(x) = \gamma \odot \left(\frac{P(x)}{\sigma(x)}\right) + \beta$$

where $P(x) = x - \bar{x}\mathbf{1}$ centers the vector and $\odot$ is elementwise multiplication.

## Operators

| Symbol | Meaning |
|--------|---------|
| $A^{i,h}$ | Attention pattern matrix for layer $i$, head $h$ |
| $\tilde{A}^i$ | Attention sub-layer with pre-norm: $\tilde{A}^i(x) = \text{Attn}^i(LN_A^i(x))$ |
| $M^i$ | MLP layer $i$ (without layer norm) |
| $\tilde{M}^i$ | MLP sub-layer with pre-norm: $\tilde{M}^i(x) = M^i(LN_M^i(x))$ |
| $B^i$ | Full transformer block $i$: $B^i(x) = x + \tilde{A}^i(x) + \tilde{M}^i(x + \tilde{A}^i(x))$ |
| $\Delta B^i$ | Block contribution operator: $\Delta B^i(x) = B^i(x) - x$ |

The tilde notation $\tilde{A}$, $\tilde{M}$ indicates sub-layers with their preceding layer normalization absorbed.

### Block Contribution Operator

The operator $\Delta B^i$ extracts just the additive contribution from block $i$:
$$\Delta B^i(x) = \tilde{A}^i(x) + \tilde{M}^i(x + \tilde{A}^i(x))$$

This can be decomposed into attention and MLP components:
$$\Delta B^i(x) = \Delta B^i_A(x) + \Delta B^i_M(x)$$

where:
- $\Delta B^i_A(x) = \tilde{A}^i(x)$ — attention contribution
- $\Delta B^i_M(x) = \tilde{M}^i(x + \tilde{A}^i(x))$ — MLP contribution

Note: The MLP sees the residual *after* attention has been added, so these terms are not independent.

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
| $x^i$ | Shorthand for $T^i(x)$ — residual after block $i$ |

**Important**: $T^i(x)$ produces the *unnormalized* residual. The full transformer applies the terminal layer norm:
$$T(x) = LN^T(T^n(x))$$

The truncated transformer composes block operators:
$$T^i = B^i \circ B^{i-1} \circ \cdots \circ B^1$$

## Expansion Identity

The residual $x^i = T^i(x)$ can be expanded as a sum of block contributions:

**Operator form:**
$$T^n(x) = x + \sum_{i=1}^{n} \Delta B^i(x^{i-1})$$

where each $\Delta B^i$ acts on the accumulated residual $x^{i-1} = T^{i-1}(x)$.

**Expanded form:**
$$T^n(x) = x + \Delta B^1(x) + \Delta B^2(x^1) + \cdots + \Delta B^n(x^{n-1})$$

The full transformer output applies the terminal layer norm:
$$T(x) = LN^T(T^n(x)) = LN^T\left(x + \sum_{i=1}^{n} \Delta B^i(x^{i-1})\right)$$

### Attention/MLP Decomposition

Each block contribution decomposes into attention and MLP components:
$$\Delta B^i(x^{i-1}) = \Delta B^i_A(x^{i-1}) + \Delta B^i_M(x^{i-1})$$

For pre-norm architectures (Pythia/GPT-NeoX with parallel attention):
- $\Delta B^i_A(x) = \tilde{A}^i(x)$ — attention contribution
- $\Delta B^i_M(x) = \tilde{M}^i(x)$ — MLP contribution (parallel, sees same input as attention)

For sequential pre-norm architectures:
- $\Delta B^i_A(x) = \tilde{A}^i(x)$
- $\Delta B^i_M(x) = \tilde{M}^i(x + \tilde{A}^i(x))$ — MLP sees post-attention residual

## Logit Decomposition

For a target token $t$ with unembedding vector $\bar{t}$, the logit is:
$$z_t = \langle \bar{t}, LN^T(x^n) \rangle$$

### Expansion Through Layer Norm

Expanding through the layer norm (see `analysis/speculation.md` for derivation):
$$z_t = \frac{1}{\sigma} \sum_{i} \langle \bar{t}_\gamma, P(x_i) \rangle + b_t$$

where:
- $x^n = \sum_i x_i$ is the expanded residual (embedding + block contributions)
- $\bar{t}_\gamma = \gamma^T \odot \bar{t}$ — scaled unembedding
- $P(x_i) = x_i - \bar{x}_i \mathbf{1}$ — mean-centered term
- $\sigma = \|P(x^n)\|/\sqrt{N}$ — standard deviation of total residual
- $b_t = \bar{t} \cdot \beta^T$ — bias contribution

This enables decomposition: each term $\langle \bar{t}_\gamma, P(x_i) \rangle / \sigma$ shows that term's contribution to the logit.

### Remainder Terms

When most contribution comes from a few terms, use remainder notation:
$$z_t \approx \frac{1}{\sigma} \left( \sum_{i \in S} \langle \bar{t}_\gamma, P(x_i) \rangle + R \right) + b_t$$

where $S$ is the set of significant terms and $R = \sum_{i \notin S} \langle \bar{t}_\gamma, P(x_i) \rangle$ collects negligible contributions.

Display as: `z_t ≈ c₁ + c₂ + ... + R` where $|R| < \epsilon |z_t|$.

## MLP Input Tracing

For sequential pre-norm architectures, the MLP at block $i$ receives:
$$\text{input to } M^i = LN_M^i(x^{i-1} + \tilde{A}^i(x^{i-1}))$$

This is the residual after $i-1$ blocks, plus the attention output from block $i$, then layer-normalized.

To trace where a significant MLP contribution came from:
1. Identify which $\Delta B^i_M$ dominates the logit
2. Examine the input: $x^{i-1} + \Delta B^i_A(x^{i-1})$
3. Recursively expand $x^{i-1}$ to trace earlier contributions

For parallel architectures (Pythia), the MLP sees the same input as attention:
$$\text{input to } M^i = LN_M^i(x^{i-1})$$

