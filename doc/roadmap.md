# TransformerAlgebra Roadmap

*January 2026*

This document captures the vision and requirements for symbolic analysis of transformer internal states, and how this drives development across the three related repositories.

## Current State

**investigate.ipynb** demonstrates symbolic logit lens analysis with named vectors:
- `PromptedTransformer` class: `T = PromptedTransformer(model, tokenizer, "The capital of Ireland")`
- Callable syntax: `T(" is")` returns a `ResidualVector` representing `T(embed(' is'))`
- Symbolic logits: `logits(x)[" Dublin"]` displays as `<unembed(' Dublin'), T(embed(' is'))> = 786.74`
- Probability predictions: `predict(x)` returns softmax probabilities with symbolic display
- Summary methods: `logits(x).summary()` shows top-k predictions

**Current output format:**
```python
>>> L = logits(T(" is"))
>>> L[" Dublin"]
<unembed(' Dublin'), T(embed(' is'))> = 786.74

>>> predict(x).summary()
"Top 5 predictions:
  1. ' the' (13.66%, logit=790.24)
  2. ' a' (4.18%, logit=789.05)
  ..."
```

This provides semantic grounding through symbolic inner products.

---

## Vision: Symbolic Output

Instead of raw logits, we want **symbolic expressions** showing inner products between named vectors:

**Desired output format:**
```
⟨Dublin̄, x²⟩ = 0.73   # angle-based, after block 2
⟨Dublin̄, LN(x^n)⟩ = 0.89   # final layer normalized

Decomposition at final layer:
  ⟨Dublin̄, embed(' is')⟩ = 0.12      # contribution from token embedding
  ⟨Dublin̄, ΔB³_A⟩ = 0.31            # contribution from attention in block 3
  ⟨Dublin̄, ΔB^n_M⟩ = 0.24           # contribution from MLP in final block
  R = 0.02                           # remainder (other terms)
```

Key requirements:
1. **Named vectors** - use token-based names (underline for embedding, overline for unembedding)
2. **Operator notation** - show block contributions as `ΔB^i` with `_A`/`_M` for sublayers
3. **Logits and probabilities** - report inner products (logits) and softmax probabilities
4. **Remainder terms** - collapse negligible contributions for clarity
5. **MLP input tracing** - when MLP dominates, trace where its input came from

---

## Core Abstractions (TransformerAlgebra Python)

### Residual
A vector with provenance: values, position, layer, symbolic label, and optionally how it was computed.

### PromptedTransformer
Model + context combination that caches all residuals and provides access to:
- Residuals by (layer, position)
- Per-block contributions (attention heads, MLP)
- Named embedding/unembedding vectors

### SymbolicInnerProduct
A displayable inner product: `⟨left, right⟩ = value` with symbolic names and optional normalization.

---

## Python vs Julia: Division of Labor

### TransformerAlgebra (Python)
**Role**: Model interface, data extraction, reference resolution

- Load HuggingFace models
- Define **reference types** that describe how to navigate the model graph
- Cache intermediate states (residuals, block contributions) per context
- Provide **TransformerService** API for Julia to call
- Handle all torch operations (forward pass, gradients)

### SymbolicTransformer (Julia)
**Role**: Symbolic manipulation, expansion rules, notation rendering

- Hold symbolic **expressions** built from references
- Apply algebraic transformations (expand, simplify)
- Call Python to resolve references when evaluation is needed
- Format output with notation-aligned display
- Use **PythonCall.jl** for direct Python interop

### Interface Design (see `doc/interface.md`)

The interface is based on **reference types** rather than raw data:

- **TokenRef** - token_id + text
- **EmbeddingRef** - token + path to W_E
- **UnembeddingRef** - token + path to W_U
- **ResidualRef** - context_id + layer + position
- **BlockContribRef** - context_id + block + position + component
- **LayerNormRef** - paths to γ, β + epsilon

Julia builds expressions from these refs and calls Python to resolve them.

---

## Immediate Next Steps (TransformerAlgebra)

### Phase 1: Richer Extraction (Python)

1. ~~**Inner product reporting**~~ ✓ DONE
   - `logits(x)[token]` returns symbolic `<unembed(token), T(x)> = value`
   - `predict(x)` computes softmax probabilities
   - `summary()` methods format top-k predictions

2. ~~**Implement `expand()`**~~ ✓ DONE (Level 1 & 2)
   - Level 1: `T^n(x)` → `x + ΔB^1(x) + ΔB^2(x^1) + ... + ΔB^n(x^{n-1})`
   - Level 2: `ΔB^i` → `ΔB^i_A + ΔB^i_M` (attention + MLP)
   - Each term is a referenceable VectorLike object
   - Sums verified mathematically exact
   - See `doc/expand_issues.md` for implementation details

3. ~~**Implement contribution analysis**~~ ✓ DONE
   - `contribution(expanded, logit)` shows how each term contributes to an inner product
   - Display as ranked list with `summary()` and `top_k()`
   - Experimental finding: final block MLP dominates (~80% of logit contribution)
   - See `doc/contribution.md` for design and results

4. **MLP input tracing** ← NEXT PRIORITY
   - Given that `ΔB^n_M` dominates, trace *where its input came from*
   - Input to MLP: `LN_M^i(x^{i-1})` (parallel) or `LN_M^i(x^{i-1} + ΔB^i_A)` (sequential)
   - Implement `mlp_input(block_contrib)` to extract the pre-MLP residual
   - Recursively expand to identify which earlier blocks/positions contributed
   - Goal: keep analysis high-level (block attribution) rather than neuron-level
   - See `doc/notation.md` § MLP Input Tracing for notation

5. **Remainder terms for concise display**
   - When showing contributions, collapse negligible terms into remainder `R`
   - Display: `z_t ≈ c₁ + c₂ + c₃ + R` where `|R| < ε|z_t|`
   - Implement `contribution(...).simplified(threshold=0.05)`

6. **Expand Level 3: Per-head attention** (lower priority)
   - `ΔB^i_A` → `Σ_h ΔB^{i,h}_A` (per-head contributions)
   - Useful for understanding which heads attend to relevant context
   - May be needed for full MLP input tracing

7. **Track position embedding contributions** (lower priority)
   - Pythia uses rotary embeddings (RoPE)—position is applied per-head in attention
   - Less straightforward than additive positional embeddings

### Phase 2: Export Format (Python → Julia)

6. **Define serialization format** - See `interface.md` for HDF5 export format and data contract

7. **Python export function**
   ```python
   lens.export("analysis.h5", prompt, include_attention=True, include_mlp=True)
   ```

---

## API Sketch (Python)

```python
from transformer_algebra import (
    PromptedTransformer, load_pythia_model, expand, logits, contribution
)

# Load model
model, tokenizer = load_pythia_model()

# Create prompted transformer (caches all residuals)
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

# Get residual for next token
x = T(" is")  # ResidualVector: T^n(embed(" is"))

# Expand into block contributions
ex = expand(x)  # x + ΔB^1 + ΔB^2 + ... + ΔB^n
print(ex)  # "embed(' is') + ΔB^0 + ΔB^1 + ... + ΔB^11"

# Analyze contributions to a specific logit
dublin_logit = logits(x)[" Dublin"]
contrib = contribution(ex, dublin_logit)
print(contrib.summary())
# "Top 5 contributions to ' Dublin':
#    ΔB^11_M: +1522.12 (+80.8%)
#    ΔB^11_A: +264.66 (+14.1%)
#    ..."

# Expand further to see attention vs MLP in final block
ex2 = ex.expand()  # Expands ΔB^i -> ΔB^i_A + ΔB^i_M

# Future: trace MLP input
# mlp_input = ex[-1].mlp_input()  # What fed into ΔB^11_M?
```

---

## Success Criteria

A successful TransformerAlgebra implementation will:

1. **Symbolic output**: Display inner products with named vectors like `⟨Dublin̄, x₅¹²⟩ = 0.89`

2. **Decomposition**: Show which attention heads and MLP layers contributed to a residual

3. **Notation consistency**: Output aligns with the mathematical notation in `doc/notation.md`

4. **Clean export**: Provide HDF5 files that SymbolicTransformer can consume

5. **Referenceable terms**: Any vector or contribution can be extracted and reused in further analysis

---

## Open Questions

1. **Normalization**: Should we always use cosine similarity, or preserve raw logits for some analyses?

2. **Approximation handling**: How to represent/track when linearization or other approximations are applied?

3. **Position encoding**: How to symbolically represent rotary position embeddings?

4. **Attention pattern storage**: Full O(n²) patterns per head are memory-intensive—store on demand?

5. **Batching**: Support for analyzing multiple prompts simultaneously?
