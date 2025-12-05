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

## Block-wise Analysis

### Isolating the First Block

The notation $T_i(c)$ for block $i$ of a prompted transformer is useful, though whether $T_2(c)$ means "block 2 alone" or "blocks 1-2 composed" may depend on context—to be resolved through examples.

For the first block, $T_1(c)(x)$ is cleanest to analyze because:
- Attention only sees context embeddings $\{x_k^0\}_{k < j}$ (fixed once context given)
- Plus the input $x$ at position $j$
- No dependence on earlier block computations at context positions

### Logit Difference Analysis

When investigating why token A was predicted over token B, define the **logit difference direction**:
$$d = \overline{A} - \overline{B}$$

Then for any contribution $\Delta x$:
$$\langle d, \Delta x \rangle > 0 \implies \text{pushes toward A over B}$$

This projects the high-dimensional residual onto a 1D "decision axis." More generally, $\{\overline{A}, \overline{B}\}$ spans a 2D subspace for investigation.

### First Block Contribution Decomposition

$$\Delta x_j^1 = \underbrace{\tilde{A}^1(x)}_{\text{attention}} + \underbrace{\tilde{M}^1(x + \tilde{A}^1(x))}_{\text{MLP}}$$

We can compute $\langle d, \tilde{A}^1(x) \rangle$ and $\langle d, \tilde{M}^1(\cdot) \rangle$ separately to see whether attention or MLP drives the prediction.

### Feature Entanglement via Layer Norm

The input to block 1 at position $k$ is $x_k^0 = \underline{t_k} + p_k$ (token + position). One might hope to separate contributions:
$$V^{1,h}(LN(x_k^0)) \stackrel{?}{\approx} V^{1,h}(LN(\underline{t_k})) + V^{1,h}(LN(p_k))$$

**But layer norm breaks linearity**: $LN(a + b) \neq LN(a) + LN(b)$

So token and position features are entangled through attention. Options:
1. Empirically measure how much each matters
2. Accept combined $x_k^0$ as the unit of analysis
3. Ablation experiments (zero out positional embeddings, etc.)

### Attention: "Where" vs "What"

The attention pattern $A^{1,h}_{jk}$ (from Q·K) determines **where** to look. The value projection $V^{1,h}$ determines **what** gets moved. These are separable computations worth analyzing independently.


## Functional Discourse Grammar

Functional Discourse Grammar is normally used to describe the production of spoken language in a way that is comparable across widely differing languages. How can our notation be informed by ideas from FDG?

The features of such notation includes:

* Describe a discourse, not just a sentence
* Utterances are described in parallel at a number of levels, which are completed in parallel as the sentence is completed.  There are links between the levels, FDG describes 4 levels, including the phonological layer which is less relevant to text processing. Lower levels follow the actual word ordering more closely, upper levels may not be completed out of order.
* Each level has a hierarchically ordered layered organization.
* Layers always have a head, and usually a modifier (both of which may themselves be layers or lexical content), and may also be specified by a linguistic-operator and carry a linguistic-function.

One challenge is the use of similar terminology with quite distinct meanings. In particular the use of layer and function in FDG may lead to confusion, we should be consistent in referring to linguistic-layer and linguistic-function when referring to these.

A key difference between transformer language models and typical FDG tasks is that FDG is focussed on the formulation of speech rather than language interpretation, while generative language models both interpret and formulate language, with an indistinct boundary between the two. I expect the first and last layers to remain close to morphosyntactic representation, and internal transformer-layers to be reflect the interpersonal and representational linguistic-levels, but don't want to pre-emptively constrain the analysis in this way.

### Methodological Constraint

Structure should emerge from analyzing specific texts through the model, not be imposed from linguistic theory. FDG informs *expectations* (stacked interpretive layers, bracketed spans, head-modifier relations) but not the encoding. 

Minimum expectations worth validating:
- Multi-token words should be identifiable as units
- Different surface forms of the same word should show relatable representations

Beyond these, let the transformer's internal organization reveal itself.

### Representing Hierarchical Structure

For describing what we find, an XML-like notation aligned to token positions:

```
Position:     0        1        2         3        4
Tokens:       The      quick    brown     fox      jumps
              ─────────────────────────────────────────
Level 1:      <NP                                  >  <VP...
Level 2:        <Det>  <Adj>    <Adj>     <N>         <V>
Level 3:                        [modifies→fox]
```

Multiple parallel "documents" stacked vertically, aligned at positions. Each level can mark:
- **Spans**: bracketed regions `<...>`
- **Heads**: the central element of a span
- **Modifiers**: elements that attach to heads, with directionality

This is purely descriptive—a way to record observations, not a claim about transformer internals.

### Connecting Representation to Analysis

The algebraic notation ($T_1(c)(x)$, attention patterns, residual decomposition) provides the *analytical tools*. The hierarchical representation provides a *descriptive language* for what we find.

The bridge between them:
1. **Attention patterns** may reveal spans—if positions 1-3 strongly attend to each other, that suggests a unit
2. **Residual similarity** may reveal heads—if position 3's residual "summarizes" positions 1-3, it's acting as head
3. **Logit projections** may reveal modifier relations—if removing position 2's contribution changes what position 3 predicts

The open question: can we formalize "position $j$ acts as head of span $[a,b]$" in terms of residual stream properties?


---

## Open Questions

1. Can we formalize the "cone of possible outputs" for an MLP more precisely?
2. Would outer products be useful for tracking which subspaces attention heads care about?
3. How does the residual stream's high dimensionality enable superposition of features?
4. Should $T_2(c)$ mean block 2 alone, or blocks 1-2 composed? (Resolve via examples)
5. Can "position $j$ is head of span $[a,b]$" be defined via attention/residual properties?
6. Do multi-token words show characteristic attention or residual patterns?
