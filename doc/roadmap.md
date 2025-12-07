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

## Core Abstractions Needed

### 1. Residual (Vector with Provenance)

A residual vector that knows its origin:
```
Residual:
  - values: torch.Tensor (d_model,)
  - position: int
  - layer: int
  - label: str  # e.g., "x₁²" or "Δx₁^(A,3,2)"
  - source: Optional[Transformation]  # how was this computed?
```

### 2. Transformation (Expression Tree)

An expression describing how a result was computed:
```
Transformation:
  - operator: str  # "attention_head", "mlp", "sum", "layer_norm", etc.
  - operands: list[Transformation | Residual]
  - parameters: dict  # layer=3, head=2, etc.
  - result: Optional[Residual]  # cached computation
```

### 3. PromptedTransformer (Model + Context)

The combination of a model with cached context state:
```
PromptedTransformer:
  - model: HuggingFace model
  - context: str
  - residuals: dict[(layer, position) -> Residual]
  - contributions: dict[(layer, position) -> list[Residual]]  # attention + MLP parts
```

### 4. SymbolicInnerProduct (Display Object)

A displayable inner product between named vectors:
```
SymbolicInnerProduct:
  - left: Residual | str  # e.g., unembed vector "Dublin̄"
  - right: Residual  
  - value: float  # computed inner product
  - normalized: bool  # cosine similarity or raw dot product
  - display() -> str  # "⟨Dublin̄, x₁²⟩ = 0.73"
```

---

## Python vs Julia: Division of Labor

After reviewing the codebase, here's the proposed split:

### TransformerAlgebra (Python)
**Role**: Model interface, data extraction, initial prototyping

Responsibilities:
- Load HuggingFace models (transformers library)
- Extract hidden states, attention patterns, MLP outputs
- Serialize intermediate states for Julia consumption
- Provide Jupyter notebook interface for exploration
- Quick iteration on new analysis ideas

### SymbolicTransformer (Julia)  
**Role**: Symbolic manipulation, notation rendering, algebraic operations

Responsibilities:
- Define `Residual`, `Transformation` types with Julia's type system
- Operator overloading for mathematical notation (`*`, `+`, `'` for transpose)
- Symbolic expression trees with expansion/simplification
- LaTeX rendering of expressions
- Macros for notation-close DSL (e.g., `@transform x₁² = Ã²(x₁¹)`)

### VectorTransformer (Julia)
**Role**: Clean reference implementation for validation

Responsibilities:
- Pure Julia transformer forward pass
- Component-by-component validation against Python
- Basis for understanding exact transformer operations
- Numerical experiments with layer normalization, attention

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

4. **Define serialization format**
   - JSON or HDF5 with:
     - All residuals indexed by (layer, position)
     - Attention patterns per head
     - MLP contributions per block
     - Token information (text, position, id)
   - Single file per prompt analysis

5. **Python export function**
   ```python
   lens.export("analysis.h5", prompt, include_attention=True, include_mlp=True)
   ```

### Phase 3: Symbolic Representation (Julia)

6. **Define core Julia types in SymbolicTransformer**
   - `struct Residual{T}` with position, layer, provenance
   - `struct Transformation` as expression tree
   - `struct InnerProduct{L,R}` for display

7. **Import from Python**
   - Load HDF5 export into Julia structures
   - Reconstruct residual graph with proper typing

8. **Symbolic operations**
   - `expand(transformation)` → show components
   - `simplify(expression)` → combine like terms
   - `latex(expression)` → render for display

---

## API Sketch

### Python (TransformerAlgebra)

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

### Julia (SymbolicTransformer)

```julia
using SymbolicTransformer

# Load exported analysis
T = load_analysis("ireland_analysis.h5")

# Same operations with Julia syntax
dublin = T.unembed[:Dublin]
x_final = T[12, 5]  # layer 12, position 5

# Inner product with operator overloading
ip = dublin' * x_final  # uses adjoint for "left side"
display(ip)  # renders ⟨D̄ublin, x₅¹²⟩ = 0.89

# Expand to see components
Δx = T.contribution(12, 5)  # block 12 contribution at position 5
expand(Δx)  # shows attention + MLP parts

# Symbolic manipulation
@transform result = Ã¹²(x₅¹¹) + M̃¹²(x₅¹¹ + Ã¹²(x₅¹¹))
simplify(result)
```

---

## Requirements for Other Repos

### SymbolicTransformer Requirements

1. **Type System**
   - [ ] `Residual{T}` parametric type with rich metadata
   - [ ] `Transformation` sum type for expression trees
   - [ ] `InnerProduct` display type with LaTeX support

2. **Operator Overloading**
   - [ ] `'` (adjoint) for transposing/conjugating vectors
   - [ ] `*` for matrix-vector and inner products  
   - [ ] `+` for residual combination with provenance tracking

3. **Import/Export**
   - [ ] HDF5 reader for Python exports
   - [ ] Reconstruct full residual graph

4. **Symbolic Operations**
   - [ ] `expand()` to decompose transformations
   - [ ] `simplify()` to combine terms
   - [ ] `latex()` for rendering

### VectorTransformer Requirements

1. **Validation**
   - [ ] Verify attention implementation matches HuggingFace
   - [ ] Verify MLP implementation matches HuggingFace
   - [ ] Verify layer norm implementation

2. **Component Extraction**
   - [ ] Function to compute single attention head output
   - [ ] Function to compute single MLP output
   - [ ] Function to compute layer norm effect

---

## Success Criteria

A successful implementation will allow:

1. **Interactive exploration**: In a notebook, expand `⟨Dublin̄, x₅¹²⟩` to see which attention heads and MLP layers contributed most

2. **Symbolic comparison**: Compare `⟨Dublin̄, x₅¹²⟩` vs `⟨Belfast̄, x₅¹²⟩` in terms of their component contributions

3. **Notation consistency**: Output reads like the mathematical notation in `doc/notation.md`

4. **Cross-validation**: Julia and Python produce the same numerical results for the same analysis

5. **Referential transparency**: Any subexpression can be extracted, named, and reused in further analysis

---

## Open Questions

1. **Normalization**: Should we always use cosine similarity, or preserve raw logits for some analyses?

2. **Approximation handling**: How to represent/track when linearization or other approximations are applied?

3. **Position encoding**: How to symbolically represent rotary position embeddings?

4. **Attention pattern storage**: Full O(n²) patterns per head are memory-intensive—store on demand?

5. **Batching**: Support for analyzing multiple prompts simultaneously?
