# Speculation

Exploratory ideas and work-in-progress thinking about transformer internals.

## Residual Space Structure

### Dual Pairing

The unembedding vectors act as **linear functionals** on the residual space. For any residual $x$:
$$
\text{logit}_{\text{token}}(x) = \langle \overline{\text{token}}, x \rangle
$$

This is a dual pairing: $\overline{\text{token}} \in V^*$ (dual space) acts on $x \in V$ (residual space). The embedding vectors $\underline{\text{token}} \in V$ live in the primal space.

While embedding/unembedding vectors don't form a basis (vocabulary size $\gg d_{\text{model}}$ typically), they define **privileged directions** for interpretation. The success of logit lens and tuned lens relies on $\langle \overline{\text{token}}, x^i_j \rangle$ being meaningful at intermediate layers.

### Feature Types in the Residual Stream

Residual vectors can be decomposed by their functional role. At position $j$, layer $i$, we can ask what directions in $x^i_j$ contribute to:

| Feature Type | Detection Method | Mathematical Signature |
|--------------|------------------|------------------------|
| **Next-token predictors** | Direct logit attribution | $\langle \overline{\text{token}}, x^i_j \rangle$ large |
| **Attention attractors** | Key-query analysis | $\langle W_K x^i_j, W_Q x^i_k \rangle$ large for $k > j$ |
| **Position influencers** | OV-circuit attribution | $W_{OV} x^i_j$ contributes to $\Delta x^{i'}_k$ for $k > j$, $i' > i$ |

These are not orthogonal—a single direction may serve multiple roles. But this decomposition helps trace *how* information flows:
1. Token → embedding → residual
2. Residual → attention attractor → copied to later position
3. Residual → predictor → logit

### Notation for Attribution

For tracing contributions:
- $x^i_j \xrightarrow{A^{i',h}} x^{i'}_k$: attention head $h$ at layer $i'$ copies from position $j$ to $k$
- $x^i_j \xrightarrow{W_U} \overline{\text{token}}$: residual contributes to token logit

This directed notation complements the algebraic expressions by showing causal flow.

---

## Prompted Transformer Structure

### Differential Context

Given the causal structure, we can ask: what does adding a token to context *change*?

$$\Delta T(t_{n+1} | c) = T(c, t_{n+1}) - T(c)$$

This "context derivative" captures how the new token modifies the transformer's behavior. Questions:
- Is there a meaningful notion of $\frac{\partial T(c)}{\partial c}$?
- How does this relate to in-context learning?

### In-Context Learning

The prompted transformer $T(c)$ can exhibit behaviors not present in bare $T$. For example, with few-shot examples in $c$, $T(c)$ performs tasks that $T$ alone cannot. This suggests:

$$T(\text{examples})(x) \approx T_{\text{fine-tuned}}(x)$$

The context state $\mathcal{C}(\text{examples})$ encodes "temporary weights" via the residual stream that attention can read.

### Context as Soft Program

We might view $T(c)$ as $T$ "programmed" by $c$:
- The bare transformer $T$ is a universal machine
- Context $c$ provides instructions/data
- $T(c)(x)$ executes the program on input $x$

This framing connects to work on transformers as universal computers and in-context learning as mesa-optimization.

---

## Geometric Decomposition of Transformer Components

### The MLP Layer: Beyond Matrix Multiplication

In Pythia (and most modern transformers), the MLP has structure:
$$
M(x) = W_{\text{out}} \cdot \sigma(W_{\text{in}} \cdot x + b_{\text{in}}) + b_{\text{out}}
$$

where $\sigma$ is GeLU. Geometric decomposition:

| Operation | Geometric Interpretation |
|-----------|-------------------------|
| $W_{\text{in}} \cdot x$ | Projection onto $d_{\text{mlp}}$ directions (typically 4× larger than $d_{\text{model}}$) |
| $\sigma(\cdot)$ | Element-wise "gating" — selectively scales each direction |
| $W_{\text{out}} \cdot (\cdot)$ | Projects back to residual space |

**Geometrically**: The MLP decomposes the input into a set of **privileged directions** (the rows of $W_{\text{in}}$), applies a **nonlinear weighting** based on how aligned the input is with each direction, then **reassembles** using $W_{\text{out}}$.

This is closer to a **"superposition of conditional projections"** than a rotation:
$$
M(x) \approx \sum_k \sigma(\langle w^{\text{in}}_k, x \rangle) \cdot w^{\text{out}}_k
$$

Each "neuron" $k$ fires proportionally to alignment with $w^{\text{in}}_k$ and contributes $w^{\text{out}}_k$ to the output.

### Why Not Pure Rotation?

Rotations (orthogonal transformations) preserve norms and angles. Neither attention nor MLP does this:
- **Attention** mixes information *across positions* — more like a convex combination than rotation
- **MLP** selectively amplifies/suppresses directions — changes norm non-uniformly

The better abstraction might be **"projective geometry"** or **"cone operations"**: the MLP maps inputs to a cone of possible outputs, with the specific output depending on where in the input space you are.

### Geometric Algebra Perspective

In geometric algebra (GA), we have:
- **Inner product** $a \cdot b$ — scalar, measures alignment
- **Outer (wedge) product** $a \wedge b$ — bivector, encodes the plane spanned by $a, b$
- **Geometric product** $ab = a \cdot b + a \wedge b$

For high-dimensional residual spaces:

| Transformer Operation | GA Analogue |
|----------------------|-------------|
| Dot product $\langle u, v \rangle$ | Inner product (grade-0) |
| Attention pattern $\text{softmax}(QK^T)$ | Collection of inner products determining mixing coefficients |
| MLP neuron activation | Inner product with threshold/gating |

**What would MLP become in GA?**

The MLP could be written as:
$$
M(x) = \sum_k g(x \cdot d_k) \, r_k
$$

where:
- $d_k \in V$ are "detector directions" (learned)
- $r_k \in V$ are "response directions" (learned)
- $g: \mathbb{R} \to \mathbb{R}$ is the activation function

This is fundamentally a **nonlinear superposition of rank-1 maps**, where the coefficient for each rank-1 map depends on the input.

### Abstract Block Decomposition

For a transformer block:

$$
B(x) = x + \underbrace{\sum_{h} \alpha_h(x) \cdot P_h(x)}_{\text{Attention: input-dependent projection}} + \underbrace{\sum_k g(x \cdot d_k) \cdot r_k}_{\text{MLP: gated superposition}}
$$

where $\alpha_h(x)$ are the attention-derived mixing weights and $P_h$ are the OV projections.

### Possible Notation Extensions

| Symbol | Meaning |
|--------|---------|
| $x \cdot d$ | Detector activation (inner product) |
| $[x]_d$ | Projection of $x$ onto direction $d$ |
| $\sum_k [x]_{d_k}^{+} r_k$ | MLP as "ReLU-gated sum" (abstracting activation) |
| $\bigvee_k d_k$ | "Feature span" — subspace spanned by detector directions |

The key insight for interpretability: **the MLP learns a discrete set of directions that are meaningful**, and the nonlinearity selects which ones to activate. This is why "neuron interpretability" sometimes works—each $(d_k, r_k)$ pair can encode a concept.

---

## Open Questions

1. Can we formalize the "cone of possible outputs" for an MLP more precisely?
2. Would outer products be useful for tracking which subspaces attention heads care about?
3. How does the residual stream's high dimensionality enable superposition of features?
