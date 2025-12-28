# Speculation

Exploratory ideas and work-in-progress thinking about transformer internals.
## Table of Contents
- [Residual Space Structure](#residual-space-structure)
- [Geometric Decomposition of Transformer Components](#geometric-decomposition-of-transformer-components)
- [Single-Token Probability Approximation](#single-token-probability-approximation)
- [Inner Product Decomposition Through Layer Norm](#inner-product-decomposition-through-layer-norm)
- [Functional Discourse Grammar](#functional-discourse-grammar)
- [Open Questions](#open-questions)

## Residual Space Structure

### Dual Pairing

Embedding vectors form a non-orthonormal, overcomplete frame for part of the residual space.
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


---


## Single-Token Probability Approximation

### The Partition Function Problem

The standard softmax formula for next-token probability is:
$$
P(\text{token} | x) = \frac{\exp(z_{\text{token}})}{\sum_{t \in V} \exp(z_t)} = \frac{\exp(z_{\text{token}})}{Z(x)}
$$

where $z_t = \langle \overline{t}, LN(x) \rangle$ and $Z(x) = \sum_{t \in V} \exp(z_t)$ is the partition function.

When analyzing contributions to a *single* token's probability, computing the full partition function over the entire vocabulary is expensive and obscures the analysis. Can we simplify?

### The Slowly-Varying Partition Function

**Claim**: When focusing on a single token, the partition function $Z(x)$ can often be treated as approximately constant, reducing probability to:
$$
P(\text{token} | x) \approx c \cdot \exp(z_{\text{token}})
$$

where $c = 1/Z(x)$ varies slowly with $x$.

**Justification**: The key observation is that:
$$
\frac{\partial \log Z}{\partial z_t} = P(t|x)
$$

This means:
- Changes to **low-probability tokens** contribute negligibly to $\log Z$
- Changes to **high-probability tokens** dominate

The partition function is a log-sum-exp—a smooth approximation to $\max$. It's dominated by the top few logits. When decomposing contributions to a mid-ranked token's logit:

1. Small perturbations to the residual change most logits proportionally
2. $\log Z$ changes slowly unless perturbations specifically target the top tokens
3. The proportionality constant $c$ absorbs the collective effect of all other tokens

### When the Approximation Holds

The approximation is useful when:
- Analyzing **why a specific token** gets its probability (e.g., "Why does ' Dublin' have probability 0.03?")
- Decomposing logit contributions across layers or components
- Comparing relative contributions without needing exact probabilities

The approximation breaks down when:
- Perturbations are large enough to **reorder top tokens** significantly
- Analysis is near **decision boundaries** where multiple tokens have similar probability
- You need **exact probability values** rather than relative contributions

### Aside: Inference Algorithms and the Full Distribution

In practice, most LLM inference algorithms don't use the full softmax distribution anyway:

| Strategy | Uses full softmax? | Description |
|----------|-------------------|-------------|
| **Greedy** | No | Always picks argmax |
| **Top-k** | No | Keep top k tokens, renormalize, sample |
| **Top-p (nucleus)** | No | Keep smallest set with cumulative prob > p |
| **Min-p** | No | Keep tokens where $P(t) > p \cdot P(\text{top})$ |
| **Temperature sampling** | Yes (at T=1) | Samples from full $P(t) \propto \exp(z_t / T)$ |

Pure temperature sampling at T=1 follows the exact softmax, but production systems typically use truncation (top-k, top-p, min-p) to avoid occasionally sampling very low-probability tokens that produce incoherent output. The model's typical behavior depends primarily on the relative ordering among top candidates, not the exact probability mass in the tail.


## Inner Product Decomposition Through Layer Norm

For decomposition across residual contributions, consider:
$$
z_{\text{token}} = \left\langle \overline{t}, \mathrm{LN}\left(\sum_i x_i\right) \right\rangle
$$

where $z_{\text{token}}$ is the logit, $\overline{t}$ is the row of the unembedding layer corresponding with the token, $\mathrm{LN}$ is layer normalisation, including $\gamma$ and $\beta$ parameters and $x = \sum_i x_i$ is the expanded residual (embedding + block contributions).

Layer normalisation with learned parameters is:
$$\mathrm{LN}(x) = \gamma \odot \left(\sqrt{N}\frac{P(x)}{|P(x)|}\right) + \beta$$

where $P(x) = x - (x \cdot \vec{1})\vec{1}$ is the mean-centering projection.

### Step 1: Separate the β term

$$z_{\text{token}} = \left\langle \bar{t}, \gamma \odot \left(\sqrt{N}\frac{P(x)}{|P(x)|}\right) \right\rangle + \langle \bar{t}, \beta \rangle$$

The second term is a constant (for fixed token), call it $b_t = \bar{t} \cdot \beta$.

### Step 2: Move γ to the query vector using the adjoint

Since Hadamard scaling is self-adjoint: $\langle u, \gamma \odot v \rangle = \langle \gamma \odot u, v \rangle$

Define $\overline{t}_\gamma = \gamma \odot \bar{t}$ (computed once for the chosen token):

$$z_{\text{token}} = \frac{\sqrt{N}}{|P(x)|} \langle \overline{t}_\gamma, P(x) \rangle + b_t$$

### Step 3: Expand the projection

$$P(x) = x - (x \cdot \vec{1})\vec{1}$$

So:
$$\langle \overline{t}_\gamma, P(x) \rangle = \langle \overline{t}_\gamma, x \rangle - (x \cdot \vec{1})(\overline{t}_\gamma \cdot \vec{1})$$

Let $\mu_{\overline{t}} = \overline{t}_\gamma \cdot \vec{1}$ (a constant for fixed token). Then:

$$\langle \overline{t}_\gamma, P(x) \rangle = \overline{t}_\gamma \cdot x - \mu_{\overline{t}}(\vec{1} \cdot x)$$

### Step 4: Expand over residual contributions

With $x = \sum_i x_i$:

$$\overline{t}_\gamma \cdot x = \sum_i (\overline{t}_\gamma \cdot x_i)$$

$$\vec{1} \cdot x = \sum_i (\vec{1} \cdot x_i)$$

So:
$$\langle \overline{t}_\gamma, P(x) \rangle = \sum_i (\overline{t}_\gamma \cdot x_i) - \mu_{\overline{t}} \sum_i (\vec{1} \cdot x_i) = \sum_i \left[ \overline{t}_\gamma \cdot x_i - \mu_{\overline{t}}(\vec{1} \cdot x_i) \right]$$

This simplifies nicely:
$$\langle \overline{t}_\gamma, P(x) \rangle = \sum_i \langle \overline{t}_\gamma, P(x_i) \rangle$$

(The projection and the inner product both distribute over sums.)

### Step 5: Handle the normalisation factor

We have:
$$|P(x)|^2 = P(x) \cdot P(x) = |x|^2 - (x \cdot \vec{1})^2$$

With $x = \sum_i x_i$:
$$|x|^2 = \sum_i |x_i|^2 + 2\sum_{i < j} x_i \cdot x_j$$

$$(x \cdot \vec{1})^2 = \left(\sum_i x_i \cdot \vec{1}\right)^2$$

This doesn't factorise nicely over the $x_i$. The normalisation couples all contributions together.

### Final expression

$$\boxed{z_{\text{token}} = \sqrt{N}\frac{\sum_i \langle \overline{t}_\gamma, P(x_i) \rangle}{\sqrt{|P(x)|^2}} + b_t}$$

where:
- $\overline{t}_\gamma = \gamma \odot \bar{t}$ (precomputed, basis-dependent but fixed)
- $b_t = \bar{t} \cdot \beta$ (constant)
- $P(x_i) = x_i - (x_i \cdot \vec{1})\vec{1}$
- $|P(x)|^2 = \left|\sum_i x_i\right|^2 - \left(\sum_i x_i \cdot \vec{1}\right)^2$

### Interpretation

The numerator decomposes cleanly as a sum of contributions from each $x_i$. Each term $\langle \overline{t}_\gamma, P(x_i) \rangle$ measures how much residual component $x_i$ contributes in the $\overline{t}_\gamma$ direction after mean-centering.

The denominator is the coupling term — it depends on the total residual and prevents clean additive decomposition. This is the fundamental non-linearity of layer norm.

For analysis, we treat $|P(x)|$ as a constant computed from the full forward pass, then contributions become linear

$$z_{\text{token}} \approx \frac{1}{\sigma} \sum_i \langle \overline{t}_\gamma, P(x_i) \rangle + b_t$$

where $\sigma = |P(x)|/\sqrt{N}$ is the standard deviation.

which is now a clean linear decomposition over residual stream contributions.

---

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
7. How much does $\log Z(x)$ actually vary across typical residual perturbations? Empirically measure the partition function stability for layer-by-layer and component-by-component decompositions.
