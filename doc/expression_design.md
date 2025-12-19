# Expression System Design

*December 2025*

This document describes the design for symbolic expressions in TransformerAlgebra, enabling `expand()` and algebraic manipulation of transformer computations.

## Goals

1. **Symbolic display** - Expressions render as mathematical notation
2. **Tensor evaluation** - Any expression can be evaluated to a concrete tensor
3. **Composability** - Expressions combine naturally (sums, inner products)
4. **Uniform interface** - `ResidualVector`, `BlockContribution`, `VectorSum` all behave alike
5. **Extensibility** - Can export to Julia/SymbolicTransformer or Python CAS for advanced manipulation

## Core Abstraction: VectorLike Protocol

All vector-valued objects implement this protocol:

```python
from typing import Protocol
import torch

class VectorLike(Protocol):
    """Protocol for vector-like objects in expressions."""

    @property
    def tensor(self) -> torch.Tensor:
        """Evaluate to concrete tensor [d_model]."""
        ...

    @property
    def d_model(self) -> int:
        """Vector dimension."""
        ...
```

Existing classes that should implement this:
- `EmbeddingVector` - token embedding
- `ResidualVector` - transformer output at a layer
- `UnembeddingVector` - token unembedding (for inner products)

New classes:
- `BlockContribution` - Δx^i from a single block
- `AttentionContribution` - attention component of Δx^i
- `MLPContribution` - MLP component of Δx^i
- `VectorSum` - sum of vector-like objects

## VectorSum: Expanded Form

A `VectorSum` is itself vector-like, enabling recursive expansion:

```python
class VectorSum:
    """Sum of vector-like objects. Is itself vector-like."""

    def __init__(self, terms: list[VectorLike], label: str = None):
        self.terms = terms
        self.label = label  # Optional compact name

    @property
    def tensor(self) -> torch.Tensor:
        return sum(t.tensor for t in self.terms)

    @property
    def d_model(self) -> int:
        return self.terms[0].d_model

    def __getitem__(self, i) -> VectorLike:
        return self.terms[i]

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def expand(self) -> "VectorSum":
        """Expand each term that can be expanded."""
        expanded = []
        for t in self.terms:
            if hasattr(t, 'expand'):
                result = t.expand()
                if isinstance(result, VectorSum):
                    expanded.extend(result.terms)
                else:
                    expanded.append(result)
            else:
                expanded.append(t)
        return VectorSum(expanded)

    def __repr__(self):
        if self.label:
            return self.label
        return " + ".join(repr(t) for t in self.terms)
```

## expand() Function

The `expand()` function transforms compact forms into sums:

```python
def expand(x: VectorLike) -> VectorSum:
    """Expand a vector into its additive components.

    For ResidualVector: T(x) -> x + Δx^0 + Δx^1 + ... + Δx^{n-1}
    For BlockContribution: Δx^i -> Δx^i_A + Δx^i_M
    For VectorSum: expand each term recursively
    """
    if hasattr(x, 'expand'):
        return x.expand()
    return VectorSum([x])  # Already atomic
```

## Expansion Levels

```
Level 0: T(embed(' is'))
         ↓ expand()
Level 1: embed(' is') + Δx^0 + Δx^1 + ... + Δx^{n-1}
         ↓ expand() on Δx^i
Level 2: embed(' is') + (Δx^0_A + Δx^0_M) + (Δx^1_A + Δx^1_M) + ...
         ↓ expand() on Δx^i_A
Level 3: embed(' is') + (Σ_h Δx^{0,h}_A + Δx^0_M) + ...
         ↓ expand() on Δx^{i,h}_A
Level 4: embed(' is') + (Σ_k A^{0,h}_{j,k} · v^{0,h}_k + ...) + ...
```

## Inner Products

Inner products between vector-like objects:

```python
class InnerProduct:
    """Scalar result of <left, right>."""

    def __init__(self, left: VectorLike, right: VectorLike):
        self.left = left
        self.right = right
        self._value = None  # Lazy evaluation

    @property
    def value(self) -> float:
        if self._value is None:
            self._value = (self.left.tensor @ self.right.tensor).item()
        return self._value

    def __repr__(self):
        return f"<{self.left}, {self.right}> = {self.value:.2f}"

    def __float__(self):
        return self.value


class InnerProductSum:
    """Sum of inner products, from distributing over VectorSum."""

    def __init__(self, terms: list[InnerProduct]):
        self.terms = terms

    @property
    def value(self) -> float:
        return sum(t.value for t in self.terms)

    def __repr__(self):
        parts = [f"<{t.left}, {t.right}>" for t in self.terms]
        return " + ".join(parts) + f" = {self.value:.2f}"
```

Distribution over sums via `__matmul__`:

```python
# In VectorSum:
def __matmul__(self, other: VectorLike) -> InnerProductSum:
    """Distribute inner product: (a + b) @ c = a@c + b@c"""
    return InnerProductSum([InnerProduct(t, other) for t in self.terms])

def __rmatmul__(self, other: VectorLike) -> InnerProductSum:
    """Distribute inner product: c @ (a + b) = c@a + c@b"""
    return InnerProductSum([InnerProduct(other, t) for t in self.terms])
```

## Usage Examples

```python
from transformer_algebra import PromptedTransformer, load_pythia_model, expand, logits

model, tokenizer = load_pythia_model()
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

# Get residual vector
x = T(" is")                    # ResidualVector: T(embed(' is'))

# Expand to layer contributions
ex = expand(x)                  # VectorSum: embed + Δx^0 + ... + Δx^{n-1}
print(ex)                       # "embed(' is') + Δx^0 + Δx^1 + ..."
print(len(ex))                  # 13 (embedding + 12 blocks)

# Access individual terms
print(ex[0])                    # embed(' is')
print(ex[1])                    # Δx^0

# Expand further
ex2 = ex[1].expand()            # VectorSum: Δx^0_A + Δx^0_M
print(ex2)                      # "Δx^0_A + Δx^0_M"

# Inner products distribute over sums
u = T.unembed(" Dublin")        # UnembeddingVector
ip = u @ ex                     # InnerProductSum
print(ip)
# "<unembed(' Dublin'), embed(' is')> + <unembed(' Dublin'), Δx^0> + ... = 786.74"

# Identify which layers contribute most
for i, term in enumerate(ip.terms):
    print(f"Layer {i}: {term.value:+.2f}")
```

## Integration with logits()

The `logits()` function should handle both concrete and expanded forms:

```python
def logits(x: VectorLike | VectorSum) -> LogitMapping | ExpandedLogitMapping:
    """Compute logits from a vector or expanded vector."""
    if isinstance(x, VectorSum):
        return ExpandedLogitMapping(x)
    else:
        # Existing behavior
        normed = x.normed
        logits_tensor = x.transformer._unembed(normed)
        return LogitMapping(logits_tensor, x)
```

## Layer Norm Handling

The final layer norm complicates expansion because:
```
logit = <u, LN(T(x))> ≠ <u, LN(x + Δx^0 + ...)>
```

Layer norm is not distributive. However, we can use the identity:
```
<u, LN(Σ aᵢ)> = (Σ <u, aᵢ>) / ||Σ aᵢ||
```

This means:
1. Individual `<u, Δx^i>` values don't sum to the final logit
2. But they show relative contributions (before normalization)
3. The normalization factor `||Σ aᵢ||` can be reported separately

Design choice: Report unnormalized contributions with the scale factor, or report the true decomposition? TBD based on interpretability needs.

## Export to Julia/CAS

For advanced symbolic manipulation, export the expression structure:

```python
class VectorSum:
    def to_dict(self) -> dict:
        """Export for Julia/SymbolicTransformer."""
        return {
            "type": "sum",
            "terms": [t.to_dict() for t in self.terms],
        }

class BlockContribution:
    def to_dict(self) -> dict:
        return {
            "type": "block_contribution",
            "layer": self.layer,
            "attention": self.attention.to_dict(),
            "mlp": self.mlp.to_dict(),
        }
```

This allows Python to extract data and Julia to handle symbolic manipulation and rendering.

## Implementation Plan

1. **Add VectorLike protocol** - Define in `core.py` or new `expressions.py`
2. **Implement BlockContribution** - Extract per-block outputs from model
3. **Implement VectorSum** - With `__getitem__`, iteration, tensor evaluation
4. **Implement expand()** - For ResidualVector → VectorSum
5. **Add inner product distribution** - `__matmul__` on VectorSum
6. **Update logits()** - Handle expanded forms
7. **Add export methods** - `to_dict()` for Julia integration

## Open Questions

1. **Layer norm handling** - Report unnormalized contributions with scale factor, or something else?

2. **Attention expansion details** - How deep should `expand()` go? Options:
   - Stop at per-head contributions
   - Go to attention weights × value vectors
   - Make depth configurable

3. **Memory/performance** - Caching strategy for expanded forms?

4. **SymPy integration** - Worth using for scalar expressions (inner products), even if not for tensors?

5. **Notation in __repr__** - Match LaTeX notation exactly, or use ASCII-friendly approximation?
