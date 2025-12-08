# TransformerAlgebra Roadmap

*December 2024*

This document captures the vision and requirements for symbolic analysis of transformer internal states, and how this drives development across the three related repositories.

## Current State

**investigate.ipynb** demonstrates basic logit lens analysis:
- Loads a Pythia model and tokenizes a prompt
- Tracks the logit for a target token (e.g., " Dublin") across layers
- Shows numeric logit values like `After block 12: +786.32`
- Lists top 5 predictions with their raw logit values

**Current output format:**
```
Logit for ' Dublin' at each layer:
  After embedding: +219.88
  After block 1: +175.46
  ...
Top 5 predictions after final layer:
  1. ' the' (logit: 786.42)
```

This is useful but opaque—numbers without semantic grounding.

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
3. **Angular representation** - cosine similarities more interpretable than raw logits
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
**Role**: Model interface, data extraction, prototyping

- Load HuggingFace models
- Extract hidden states, attention patterns, MLP outputs
- Export intermediate states in a defined format
- Jupyter notebook interface for exploration

### SymbolicTransformer (Julia)  
**Role**: Symbolic manipulation, notation rendering, validation

- Import data exported from TransformerAlgebra
- Symbolic representation and algebraic operations
- Notation-aligned display
- Reference implementation for numerical validation

---

## Immediate Next Steps (TransformerAlgebra)

### Phase 1: Richer Extraction (Python)

1. **Extract per-block contributions**
   - Add method to get attention output per head: `get_attention_contributions()`
   - Add method to get MLP output per block: `get_mlp_contributions()`
   - Cache all intermediate residuals with position/layer metadata

2. **Named vector registry**
   - Create `EmbeddingSpace` class that maps token strings to embedding vectors
   - Create `UnembeddingSpace` class for token → unembed vector
   - Support lookup by token text: `embed["Dublin"]`, `unembed["Dublin"]`

3. **Inner product reporting**
   - Compute `⟨unembed[token], residual⟩` for each layer/position
   - Normalize to cosine similarity
   - Format output with symbolic names

### Phase 2: Export Format (Python → Julia)

4. **Define serialization format** (see Interface Contract below)

5. **Python export function**
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
