# Expand Function: Implementation Status and Known Issues

*December 2025*

## Implementation Status

The `expand()` function has been implemented with two levels of expansion:

### Level 1: ResidualVector -> Block Contributions (Working)
```python
expand(T(" is"))  # Returns: embed(' is') + Δx^0 + Δx^1 + ... + Δx^{n-1}
```
- **Status**: Fully working
- **Verified**: `test_expand_residual_to_blocks` passes
- Sum of terms equals original residual tensor

### Level 2: BlockContribution -> Attention + MLP (Partially Working)
```python
block_contrib.expand()  # Returns: Δx^i_A + Δx^i_M
```
- **Status**: Individual block expansion works correctly
- **Verified**: `test_expand_block_to_attention_mlp` passes
- For a single block, attention + MLP contribution equals the block contribution

## Known Issues

### Issue 1: Multi-Level Expansion Sum Discrepancy

**Symptom**: When expanding all block contributions to Level 2 and summing, the result does not equal the original residual.

**Test**: `test_multi_level_expand` fails with assertion:
```python
assert torch.allclose(level2.tensor, x.tensor, atol=1e-5)  # Fails
```

**Observed Values**:
- `level2.tensor`: values around ~1.2 (similar to embedding scale)
- `x.tensor`: values around ~52 (full residual scale)

**What Works**:
- Level 1 expansion sums correctly
- Individual block expansions sum correctly
- Structure is correct (1 embedding + 2*n_layers attention/MLP terms)

**Suspected Causes**:

1. **Hidden State Reference Issue**: When `VectorSum.expand()` iterates over BlockContribution terms, each BlockContribution may be computing attention/MLP contributions using shared hidden states that get modified or incorrectly indexed.

2. **Position Indexing**: The `-1` position index may behave differently when computing sublayer contributions vs. when the block contributions were originally computed.

3. **Residual vs. Extended Context Mismatch**: `T(" is")` runs on an extended prompt and stores hidden states from that run. When computing sublayer contributions, we use those same hidden states, but there may be subtle differences in how positions are handled.

### Issue 2: GPT-NeoX Specific Implementation

The current implementation is hardcoded for GPT-NeoX (Pythia) architecture:
- Uses `model.gpt_neox.layers[layer]` to access blocks
- Uses `block.input_layernorm` and `block.post_attention_layernorm`
- Uses `model.gpt_neox.rotary_emb` for position embeddings

**Impact**: Will not work with other model architectures (GPT-2, LLaMA, etc.)

**Future Work**: Consider architecture abstraction or architecture-specific handlers.

### Issue 3: Attention Mask Handling

The current implementation uses a simple all-ones attention mask:
```python
attention_mask = torch.ones(1, seq_len, device=residual_before.device)
```

This may not correctly handle:
- Causal masking (though GPT-NeoX may handle this internally)
- Padding tokens
- Custom attention patterns

## Debugging Recommendations

To investigate the multi-level expansion issue:

1. **Add detailed logging in `_compute_sublayer_contributions`**:
   - Print input residual norms
   - Print attention/MLP output norms
   - Compare with expected block contribution norm

2. **Test with single-token prompts**:
   - Simpler position handling
   - No cross-token attention complications

3. **Compare layer-by-layer**:
   - Expand one block at a time
   - Check cumulative sum after each expansion
   - Identify which layer(s) introduce the discrepancy

4. **Verify hidden state indexing**:
   - Print `hidden_states[layer].shape` for each layer
   - Ensure position indexing is consistent

## Files Modified

- `src/transformer_algebra/core.py`:
  - Added `AttentionContribution` class
  - Added `MLPContribution` class
  - Updated `BlockContribution` with `expand()` method and hidden states storage
  - Added `_compute_sublayer_contributions()` to `PromptedTransformer`

- `src/transformer_algebra/__init__.py`:
  - Exported `AttentionContribution` and `MLPContribution`

- `tests/test_basic.py`:
  - Added `test_vector_class_imports` for new classes
  - Added `test_expand_residual_to_blocks` (passes)
  - Added `test_expand_block_to_attention_mlp` (passes)
  - Added `test_multi_level_expand` (fails - known issue)
  - Added `test_expand_repr` (passes)

## API Usage

```python
from transformer_algebra import (
    PromptedTransformer, load_pythia_model, expand,
    BlockContribution, AttentionContribution, MLPContribution
)

# Load model
model, tokenizer = load_pythia_model()
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

# Get residual
x = T(" is")

# Level 1 expansion
level1 = expand(x)
print(level1)  # embed(' is') + Δx^0 + Δx^1 + ...

# Access individual block contribution
block0 = level1[1]  # First block contribution

# Level 2 expansion of a single block (works correctly)
block0_expanded = block0.expand()
print(block0_expanded)  # Δx^0_A + Δx^0_M

# Verify sum equals original block
assert torch.allclose(block0_expanded.tensor, block0.tensor, atol=1e-5)
```

## How to Continue Investigating

### Reproducing the Issue

```bash
cd TransformerAlgebra
source .venv/bin/activate
python -m pytest tests/test_basic.py::TestModelLoading::test_multi_level_expand -v --run-slow
```

### Quick Debug Script

```python
from transformer_algebra import load_pythia_model, PromptedTransformer, expand
import torch

model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
T = PromptedTransformer(model, tokenizer, "Hello")
x = T(" world")

# Level 1 - should work
level1 = expand(x)
print(f"Level 1 sum matches: {torch.allclose(level1.tensor, x.tensor, atol=1e-5)}")

# Check each block individually
for i, term in enumerate(level1.terms):
    if hasattr(term, 'expand'):
        block_exp = term.expand()
        matches = torch.allclose(block_exp.tensor, term.tensor, atol=1e-5)
        print(f"Block {i-1} expand matches: {matches}")
        if not matches:
            diff = (block_exp.tensor - term.tensor).abs().max()
            print(f"  Max diff: {diff.item():.6f}")

# Level 2
level2 = level1.expand()
print(f"\nLevel 2 sum matches: {torch.allclose(level2.tensor, x.tensor, atol=1e-5)}")
print(f"Level 2 sum norm: {level2.tensor.norm():.2f}")
print(f"Original norm: {x.tensor.norm():.2f}")
```

### Key Hypothesis

The most likely cause is that when computing sublayer contributions for block `i`, we need to use the hidden state **at the input to block i**, but the position indexing or hidden state selection may be off when processing multiple blocks in sequence.

### What's Working

- Single block expansion: `block_contrib.expand()` correctly returns attention + MLP that sum to the block contribution
- Level 1 expansion: `expand(residual)` correctly returns embedding + blocks that sum to residual
- The structure of multi-level expansion is correct (right number of terms, right types)

### What's Broken

- Cumulative sum of Level 2 terms does not equal original residual
- Discrepancy is large (values ~1 vs ~50), not a numerical precision issue

## Next Steps

1. Debug and fix the multi-level expansion sum issue
2. Add architecture abstraction for non-GPT-NeoX models
3. Implement Level 3 expansion: per-head attention contributions
4. Add inner product distribution over VectorSum
