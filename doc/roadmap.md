# TransformerAlgebra Roadmap

*December 2025cd*

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
⟨Dublin̄, x₁²⟩ = 0.73   # angle-based, after block 2, position 1
⟨Dublin̄, LN(x₁¹²)⟩ = 0.89   # final layer normalized

Decomposition at final layer:
  ⟨Dublin̄, the̲₁⟩ = 0.12      # contribution from "the" embedding
  ⟨Dublin̄, Δx₁^(A,3,2)⟩ = 0.31  # contribution from attention head 3 in block 2
  ⟨Dublin̄, Δx₁^(M,7)⟩ = 0.24    # contribution from MLP in block 7
```

Key requirements:
1. **Named vectors** - use token-based names (underline for embedding, overline for unembedding)
2. **Operator labels** - show which block/layer/head produced each contribution  
3. **Logits and probabilities** - report inner products (logits) and softmax probabilities
4. **Referenceable terms** - be able to grab `Δx₁^(A,3,2)` and examine it further

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

2. **Implement `expand()`** ← NEXT
   - Converts between compact and expanded forms (see `doc/notation.md`):
     ```
     Compact:  LN(T(embed(' is')))
     Expanded: LN(embed(' is') + Δx^0 + Δx^1 + ... + Δx^{n-1})
     ```
   - Each Δx^i = T^{i+1}(x) - T^i(x) is the contribution from block i
   - Requires extracting per-block contributions (attention + MLP)
   - Each term should be a referenceable object for further analysis
   - Key relation: `<x, LN(a + b)> = (<x,a> + <x,b>) / ||a+b||`

3. **Implement attribution**
   - Given an expanded vector sum and a target token, show how each term contributes to the token's likelihood
   - Compute inner product of each term with the unembedding vector: `⟨token̄, Δx^i⟩`
   - Display attribution as a ranked list or table showing contribution magnitudes
   - Enable questions like "which block most increased/decreased the probability of this token?"
   - See `doc/attribution.md` for detailed design

4. **Extract per-block contributions** (supports expand and attribution)
   - Add method to get attention output per head: `get_attention_contributions()`
   - Add method to get MLP output per block: `get_mlp_contributions()`
   - Cache all intermediate residuals with position/layer metadata

5. **Named vector registry** (lower priority)
   - Optional: `embed["Dublin"]`, `unembed["Dublin"]` syntax
   - Current subscripting via `logits(x)[token]` may be sufficient

### Phase 2: Export Format (Python → Julia)

6. **Define serialization format** (see Interface Contract below)

7. **Python export function**
   ```python
   lens.export("analysis.h5", prompt, include_attention=True, include_mlp=True)
   ```

---

## Interface Contract: What TransformerAlgebra Exports

SymbolicTransformer will consume data exported by TransformerAlgebra. The export format provides:

### Required Data

| Field | Shape | Description |
|-------|-------|-------------|
| `residuals` | (layers+1, positions, d_model) | All residual vectors |
| `attention_contributions` | (layers, heads, positions, d_model) | Per-head output at each position |
| `mlp_contributions` | (layers, positions, d_model) | MLP output at each position |
| `tokens` | (positions,) | Token strings |
| `token_ids` | (positions,) | Token IDs |

### Optional Data (large, load on demand)

| Field | Shape | Description |
|-------|-------|-------------|
| `embeddings` | (vocab_size, d_model) | Full embedding matrix |
| `unembeddings` | (vocab_size, d_model) | Full unembedding matrix |
| `attention_patterns` | (layers, heads, positions, positions) | Full attention weights |

### Metadata

- Model name and config (layers, heads, d_model)
- Prompt text
- Indexing conventions (0-based, dimension order)

### Format

HDF5 for efficient cross-language tensor storage.

---

## What TransformerAlgebra Needs FROM SymbolicTransformer

TransformerAlgebra expects SymbolicTransformer to provide:

1. **Import capability** - Load the HDF5 export format defined above
2. **Named vector display** - Vectors labeled with token-based names (e.g., `Dublin̄`, `x₅¹²`)
3. **Inner product display** - Format `⟨left, right⟩ = value` with symbolic names
4. **Decomposition view** - Given a residual, show its components with operator/layer labels

Implementation details are the responsibility of SymbolicTransformer.

---

## API Sketch (Python)

```python
from transformer_algebra import PromptedTransformer, load_pythia

# Load model
model = load_pythia("EleutherAI/pythia-160m-deduped")

# Create prompted transformer (caches all residuals)
T = PromptedTransformer(model, "The capital of Ireland is")

# Access named vectors
dublin = T.unembed["Dublin"]  # Residual with label "Dublin̄"
x_final = T.residual(layer=-1, position=-1)  # x₅¹²

# Compute symbolic inner product
ip = dublin @ x_final  # InnerProduct object
print(ip)  # "⟨Dublin̄, x₅¹²⟩ = 0.89"

# Decompose the residual
for component in T.decompose(x_final):
    print(dublin @ component)
# "⟨Dublin̄, is̲₅⟩ = 0.12"
# "⟨Dublin̄, Δx₅^(A,3,2)⟩ = 0.31"
# "⟨Dublin̄, Δx₅^(M,7)⟩ = 0.24"

# Export for Julia
T.export("ireland_analysis.h5")
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
