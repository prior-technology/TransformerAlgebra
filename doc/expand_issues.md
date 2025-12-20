# Expand Function: Implementation Status

*December 2025*

## Summary

The `expand()` function is now fully working for two levels of expansion:

- **Level 1**: `T(x)` → `embed(x) + Δx^0 + Δx^1 + ... + Δx^{n-1}`
- **Level 2**: `Δx^i` → `Δx^i_A + Δx^i_M` (attention + MLP contributions)

All sums are mathematically exact - the expanded terms sum to the original tensor.

## Key Design Decision: Pre-LN Residuals

The transformer operator `T^i(x)` returns the **pre-final-layer-norm** residual after running `i` blocks. This is essential for correct block contribution semantics.

### Why This Matters

HuggingFace GPT-NeoX returns `hidden_states[n_layers]` as **post** `final_layer_norm`, which caused a bug where the last block's "contribution" absorbed the entire layer norm transformation (norm ~550 vs ~2 for other blocks).

We now compute the true pre-LN residual by running the last block manually, giving consistent block contributions at similar scales.

### Semantics

```python
T = PromptedTransformer(model, tokenizer, "Hello")
x = T(" world")      # Pre-LN residual: T^n(embed(" world"))
x.tensor             # The raw residual vector (pre-LN)
x.normed             # Apply final_layer_norm: LN(T^n(embed(" world")))
logits(x)            # Computes <unembed(token), LN(T(x))>
```

## Expansion Levels

```
Level 0: T(embed(' is'))
         ↓ expand()
Level 1: embed(' is') + Δx^0 + Δx^1 + ... + Δx^{n-1}
         ↓ expand() on Δx^i
Level 2: embed(' is') + (Δx^0_A + Δx^0_M) + (Δx^1_A + Δx^1_M) + ...
```

All levels verified: sums equal original tensor.

## API Usage

```python
from transformer_algebra import (
    PromptedTransformer, load_pythia_model, expand, logits
)

model, tokenizer = load_pythia_model()
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

# Get residual (pre-LN)
x = T(" is")

# Level 1 expansion
level1 = expand(x)
print(level1)  # embed(' is') + Δx^0 + Δx^1 + ...

# Level 2 expansion
level2 = level1.expand()
print(level2)  # embed(' is') + Δx^0_A + Δx^0_M + Δx^1_A + ...

# Both sums equal original
assert torch.allclose(level1.tensor, x.tensor)
assert torch.allclose(level2.tensor, x.tensor)

# Logits use the normed (post-LN) residual
L = logits(x)
print(L[" Dublin"])  # <unembed(' Dublin'), LN(T(x))> = value
```

## Files Modified

- `src/transformer_algebra/core.py`:
  - Added `_compute_pre_ln_hidden_states()` method
  - Updated `__call__` to store pre-LN residuals
  - Added `AttentionContribution` and `MLPContribution` classes
  - Added `expand()` method to `BlockContribution`

## Future Work

1. **Level 3 expansion**: Per-head attention contributions (`Δx^{i,h}_A`)
2. **Architecture abstraction**: Support for non-GPT-NeoX models
3. **Inner product distribution**: `unembed @ expanded` returning per-term contributions
