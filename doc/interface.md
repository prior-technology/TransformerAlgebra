# TransformerAlgebra ⟷ SymbolicTransformer Interface

*December 2025*

This document defines the interface between TransformerAlgebra (Python) and SymbolicTransformer (Julia) for symbolic transformer analysis.

## Design Philosophy

The interface is built on **reference types** rather than raw data. Julia holds symbolic references that describe *how* to retrieve data from the model, and calls Python to resolve them when needed. This enables:

1. **Lazy evaluation** - Data is fetched only when needed for computation
2. **Symbolic manipulation** - Julia can transform expressions without touching actual tensors
3. **Model independence** - The same reference types work across different HuggingFace architectures
4. **Gradient access** - Python can use torch autograd to compute gradients on demand

---

## Core Reference Types

These types describe locations in the model rather than containing actual data.

### ModelPath

A path through the HuggingFace model object graph.

```python
@dataclass
class ModelPath:
    """Path to a tensor in a HuggingFace model.

    Example paths for GPT-NeoX (Pythia):
        - Embedding weights: ["gpt_neox", "embed_in", "weight"]
        - Unembedding weights: ["embed_out", "weight"]
        - Final layer norm weight: ["gpt_neox", "final_layer_norm", "weight"]
        - Final layer norm bias: ["gpt_neox", "final_layer_norm", "bias"]
        - Block 5 attention output: ["gpt_neox", "layers", 5, "attention", "dense", "weight"]
    """
    segments: list[str | int]

    def resolve(self, model) -> torch.Tensor:
        """Navigate the model object graph to get the tensor."""
        obj = model
        for seg in self.segments:
            if isinstance(seg, int):
                obj = obj[seg]
            else:
                obj = getattr(obj, seg)
        return obj
```

### TokenRef

Reference to a token in the vocabulary.

```python
@dataclass
class TokenRef:
    """Reference to a token.

    Attributes:
        token_id: Index in vocabulary
        text: String representation (e.g., " Dublin" with leading space)
    """
    token_id: int
    text: str
```

### EmbeddingRef

Reference to a token embedding vector (row of W_E).

```python
@dataclass
class EmbeddingRef:
    """Reference to an embedding vector.

    Notation: underlined token, e.g., $\underline{\text{Dublin}}$
    """
    token: TokenRef
    weights_path: ModelPath  # Path to embedding matrix

    def resolve(self, model) -> torch.Tensor:
        W_E = self.weights_path.resolve(model)
        return W_E[self.token.token_id, :]
```

### UnembeddingRef

Reference to a token unembedding vector (row of W_U).

```python
@dataclass
class UnembeddingRef:
    """Reference to an unembedding vector.

    Notation: overlined token, e.g., $\overline{\text{Dublin}}$
    """
    token: TokenRef
    weights_path: ModelPath  # Path to unembedding matrix

    def resolve(self, model) -> torch.Tensor:
        W_U = self.weights_path.resolve(model)
        return W_U[self.token.token_id, :]
```

### ResidualRef

Reference to a residual vector in a specific context.

```python
@dataclass
class ResidualRef:
    """Reference to a residual vector in context.

    Notation: $x^i_j$ = residual after layer i at position j
    """
    context_id: str      # Hash/ID of the prompt context
    layer: int           # 0 = after embedding, 1..n = after blocks
    position: int        # Token position in sequence

    # Resolved lazily from cached context states
```

### LayerNormRef

Reference to layer normalization parameters.

```python
@dataclass
class LayerNormRef:
    """Reference to layer normalization.

    LN(x) = γ * (x - μ) / σ + β
    """
    weight_path: ModelPath  # Path to γ (scale)
    bias_path: ModelPath    # Path to β (shift)
    epsilon: float = 1e-5

    def resolve_params(self, model) -> tuple[torch.Tensor, torch.Tensor]:
        γ = self.weight_path.resolve(model)
        β = self.bias_path.resolve(model)
        return γ, β
```

### BlockContribRef

Reference to a transformer block's contribution to the residual stream.

```python
@dataclass
class BlockContribRef:
    """Reference to a block's contribution.

    Notation: $\Delta x^i_j$ = contribution from block i at position j
    """
    context_id: str
    block: int           # Block index (1..n)
    position: int
    component: str       # "attention", "mlp", or "total"
    head: int | None = None  # For attention, which head (None = all heads summed)
```

---

## Architecture Profiles

Different model architectures have different internal structures. The interface uses "profiles" to abstract these differences.

```python
@dataclass
class ArchitectureProfile:
    """Maps abstract concepts to concrete model paths.

    This allows the same interface to work across GPT-NeoX, Llama, etc.
    """
    name: str

    # Embedding/unembedding
    embedding_weights: ModelPath
    unembedding_weights: ModelPath

    # Final layer norm (applied before unembedding)
    final_ln_weight: ModelPath
    final_ln_bias: ModelPath
    final_ln_epsilon: float

    # Block structure
    n_blocks: int
    block_ln_weight: Callable[[int], ModelPath]   # layer -> path
    block_ln_bias: Callable[[int], ModelPath]
    attention_output: Callable[[int, int], ModelPath]  # layer, head -> path
    mlp_output: Callable[[int], ModelPath]        # layer -> path


GPT_NEOX_PROFILE = ArchitectureProfile(
    name="gpt_neox",
    embedding_weights=ModelPath(["gpt_neox", "embed_in", "weight"]),
    unembedding_weights=ModelPath(["embed_out", "weight"]),
    final_ln_weight=ModelPath(["gpt_neox", "final_layer_norm", "weight"]),
    final_ln_bias=ModelPath(["gpt_neox", "final_layer_norm", "bias"]),
    final_ln_epsilon=1e-5,
    n_blocks=12,  # For pythia-160m
    block_ln_weight=lambda i: ModelPath(["gpt_neox", "layers", i, "input_layernorm", "weight"]),
    block_ln_bias=lambda i: ModelPath(["gpt_neox", "layers", i, "input_layernorm", "bias"]),
    attention_output=lambda i, h: ModelPath(["gpt_neox", "layers", i, "attention", "dense", "weight"]),
    mlp_output=lambda i: ModelPath(["gpt_neox", "layers", i, "mlp", "dense_4h_to_h", "weight"]),
)
```

---

## Context Cache

When a prompt is run through the model, all intermediate states are cached and assigned a context ID.

```python
@dataclass
class ContextCache:
    """Cached intermediate states for a specific prompt.

    Created when PromptedTransformer runs the model.
    """
    context_id: str                           # Unique identifier
    prompt: str                               # Original prompt text
    tokens: list[TokenRef]                    # Tokenized prompt

    residuals: dict[tuple[int, int], torch.Tensor]     # (layer, pos) -> vector
    attention_contribs: dict[tuple[int, int, int], torch.Tensor]  # (layer, head, pos) -> vector
    mlp_contribs: dict[tuple[int, int], torch.Tensor]  # (layer, pos) -> vector

    # Attention patterns (optional, memory-intensive)
    attention_patterns: dict[tuple[int, int], torch.Tensor] | None  # (layer, head) -> (seq, seq)
```

---

## Python Service API

Python exposes these operations for Julia to call. This can be via:
- PythonCall.jl for direct interop
- JSON-RPC over a socket
- Shared HDF5 files

### Core Operations

```python
class TransformerService:
    """Service that Julia calls to interact with the model."""

    def __init__(self, model, tokenizer, profile: ArchitectureProfile):
        self.model = model
        self.tokenizer = tokenizer
        self.profile = profile
        self.contexts: dict[str, ContextCache] = {}

    # === Context Management ===

    def create_context(self, prompt: str) -> str:
        """Run model on prompt, cache all states, return context_id."""
        ...

    def get_context_info(self, context_id: str) -> dict:
        """Get metadata about a context (tokens, shapes, etc.)."""
        ...

    # === Resolution ===

    def resolve_embedding(self, ref: EmbeddingRef) -> list[float]:
        """Get embedding vector for a token."""
        ...

    def resolve_unembedding(self, ref: UnembeddingRef) -> list[float]:
        """Get unembedding vector for a token."""
        ...

    def resolve_residual(self, ref: ResidualRef) -> list[float]:
        """Get cached residual vector."""
        ...

    def resolve_block_contrib(self, ref: BlockContribRef) -> list[float]:
        """Get cached block contribution."""
        ...

    def resolve_layer_norm(self, ref: LayerNormRef) -> dict:
        """Get layer norm parameters {weight, bias, epsilon}."""
        ...

    # === Computation ===

    def inner_product(self, ref1, ref2) -> float:
        """Compute ⟨v1, v2⟩ for two vector refs."""
        ...

    def apply_layer_norm(self, ln_ref: LayerNormRef, vector_ref) -> list[float]:
        """Apply layer normalization to a vector."""
        ...

    def compute_logits(self, residual_ref: ResidualRef) -> list[float]:
        """Apply final LN + unembedding to get all logits."""
        ...

    def top_predictions(self, residual_ref: ResidualRef, k: int = 10) -> list[dict]:
        """Get top k predictions with tokens and logits."""
        ...

    # === Gradient Operations ===

    def gradient(self, output_ref, input_ref) -> list[float]:
        """Compute ∂output/∂input using torch autograd.

        Useful for understanding how changes propagate through the model.
        """
        ...
```

---

## Julia Expression Types

Julia builds symbolic expressions from references, enabling algebraic manipulation before evaluation.

```julia
# Base types
abstract type Expr end
abstract type VectorExpr <: Expr end
abstract type ScalarExpr <: Expr end

# Leaf expressions (wrap references from Python)
struct EmbeddingExpr <: VectorExpr
    ref::Dict  # Serialized EmbeddingRef
    label::String  # e.g., "Dublin" for display
end

struct UnembeddingExpr <: VectorExpr
    ref::Dict
    label::String
end

struct ResidualExpr <: VectorExpr
    ref::Dict
    label::String  # e.g., "x₅¹²"
end

struct BlockContribExpr <: VectorExpr
    ref::Dict
    label::String  # e.g., "Δx₅^(A,3)"
end

# Compound expressions
struct SumExpr <: VectorExpr
    terms::Vector{VectorExpr}
end

struct ScaledExpr <: VectorExpr
    scalar::Union{Float64, ScalarExpr}
    vector::VectorExpr
end

struct LayerNormExpr <: VectorExpr
    ln_ref::Dict
    input::VectorExpr
end

struct InnerProductExpr <: ScalarExpr
    left::VectorExpr
    right::VectorExpr
end

# Operations
Base.:(+)(a::VectorExpr, b::VectorExpr) = SumExpr([a, b])
Base.:(*)(s::Number, v::VectorExpr) = ScaledExpr(s, v)
LinearAlgebra.dot(a::VectorExpr, b::VectorExpr) = InnerProductExpr(a, b)
```

---

## Expansion Rules

The key algebraic operation is **expansion** - decomposing an expression into more primitive terms.

### Residual Expansion

A final residual can be expanded into embedding + block contributions:

$$x^L_j = \underline{t_j} + \sum_{i=1}^{L} \Delta x^i_j$$

```julia
function expand(r::ResidualExpr)
    # Get context info from Python
    ctx = get_context_info(r.ref["context_id"])

    # Build sum of embedding + all block contributions
    embedding = EmbeddingExpr(
        Dict("token" => ctx.tokens[r.ref["position"]], ...),
        ctx.tokens[r.ref["position"]].text
    )

    contribs = [BlockContribExpr(...) for i in 1:ctx.n_layers]

    return SumExpr([embedding, contribs...])
end
```

### Logit Expansion (Inner Product)

A logit is an inner product that can be expanded through layer norm:

$$\langle \bar{t}, LN(x) \rangle = \frac{1}{\|x - \mu\|} \sum_i \langle \bar{t} \odot \gamma, x_i - \mu_i \rangle + \langle \bar{t}, \beta \rangle$$

The layer norm expansion uses the relation:
$$\langle y, LN(a + b) \rangle = \frac{\langle y, a - \mu_a \rangle + \langle y, b - \mu_b \rangle}{\|a + b - \mu_{a+b}\|}$$

```julia
function expand(ip::InnerProductExpr)
    if ip.right isa LayerNormExpr && ip.right.input isa SumExpr
        # Distribute inner product through layer norm
        ln = ip.right
        terms = ln.input.terms

        # Compute scale factor (requires Python call)
        scale = compute_ln_scale(ln.ln_ref, terms)

        # Each term becomes a separate inner product
        expanded_terms = [
            ScaledExpr(scale, InnerProductExpr(
                scale_by_gamma(ip.left, ln.ln_ref),
                center(term)
            ))
            for term in terms
        ]

        # Add bias term
        push!(expanded_terms, InnerProductExpr(ip.left, BiasExpr(ln.ln_ref)))

        return SumExpr(expanded_terms)
    end
    return ip  # Can't expand further
end
```

---

## Serialization Format

For communication between Python and Julia (when not using PythonCall directly).

### JSON Schema

```json
{
  "type": "EmbeddingRef",
  "token": {"token_id": 12876, "text": " Dublin"},
  "weights_path": ["gpt_neox", "embed_in", "weight"]
}
```

```json
{
  "type": "ResidualRef",
  "context_id": "abc123",
  "layer": 12,
  "position": 5
}
```

```json
{
  "type": "InnerProductExpr",
  "left": {"type": "UnembeddingRef", ...},
  "right": {"type": "LayerNormExpr", "input": {"type": "SumExpr", ...}}
}
```

### HDF5 for Bulk Data

When Julia needs actual tensor values, Python can write to HDF5:

```
/context_{id}/
    /metadata          # JSON: tokens, config, etc.
    /residuals         # (n_layers+1, n_positions, d_model)
    /attention_contribs # (n_layers, n_heads, n_positions, d_model)
    /mlp_contribs      # (n_layers, n_positions, d_model)
```

---

## Example Workflow

1. **User runs notebook in Python:**
   ```python
   T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
   x = T(" is")  # Returns ResidualRef
   ```

2. **User asks for logit decomposition:**
   ```python
   dublin_logit = logits(x)["Dublin"]  # Returns InnerProductExpr
   expanded = expand(dublin_logit)      # Julia computes symbolic expansion
   ```

3. **Julia builds symbolic tree:**
   ```
   InnerProduct(
     UnembeddingExpr("Dublin"),
     LayerNormExpr(
       SumExpr([
         EmbeddingExpr(" is"),
         BlockContribExpr(layer=1, ...),
         BlockContribExpr(layer=2, ...),
         ...
       ])
     )
   )
   ```

4. **Julia expands through layer norm:**
   ```
   SumExpr([
     scale * InnerProduct(γ ⊙ Dublin̄, center(is̲)),
     scale * InnerProduct(γ ⊙ Dublin̄, center(Δx¹)),
     scale * InnerProduct(γ ⊙ Dublin̄, center(Δx²)),
     ...
     InnerProduct(Dublin̄, β)
   ])
   ```

5. **Julia calls Python to evaluate each term:**
   ```julia
   for term in expanded.terms
       value = evaluate(python_service, term)
       println("$(display(term)) = $value")
   end
   ```

6. **Output:**
   ```
   ⟨Dublin̄, is̲⟩ = 0.12
   ⟨Dublin̄, Δx¹⟩ = 0.03
   ⟨Dublin̄, Δx⁷⟩ = 0.31   # <-- Biggest contribution from block 7
   ...
   ```

---

## Open Design Questions

1. **PythonCall vs RPC**: Direct Julia-Python interop is simpler but couples the implementations. RPC/HDF5 allows independent processes.

2. **Caching strategy**: Should Julia cache resolved vectors, or always call Python? Memory vs latency tradeoff.

3. **Gradient integration**: How deeply to integrate torch autograd? Could enable symbolic differentiation in Julia.

4. **Expression normalization**: Should expressions be canonicalized (e.g., sorted sums)? Helps with comparison and caching.

5. **Position encoding**: Rotary embeddings don't have a simple additive decomposition. How to represent symbolically?
