# Contribution Analysis

The `contribution` function shows how each term in an expanded expression contributes to the logit (and thus probability) of a target token.

## Notation

We define `▷` as the contribution operator. If `x = Σᵢ xᵢ` (an expanded residual):

```
x ▷ t = { x₀: 20%, x₁: 2%, x₂: 50%, ... }
```

reads as: "for token t, term x₀ contributes 20% of the logit, x₁ contributes 2%, etc."

## API

The contribution function acts on an expanded residual and an inner product (like a logit):

```python
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
x = T(" is")                           # Pre-LN residual
ex = expand(x)                         # Expand to block contributions
dublin_logit = logits(x)[" Dublin"]    # Inner product ⟨unembed(" Dublin"), LN(x)⟩
contrib = contribution(ex, dublin_logit)
print(contrib.summary())
```

The second argument is a `LogitValue` from `logits(x)[token]`, which represents the inner product `⟨unembed(token), LN(x)⟩`. This design allows contribution analysis for any layer-normalized inner product, not just logits.

## Why This Works

See `analysis/speculation.md` § "Single-Token Probability Approximation" and "Logit Decomposition Through Layer Norm" for the mathematical justification.

**Summary:**
1. The partition function Z(x) varies slowly, so `log P(t) ≈ logit(t) - const`
2. The logit decomposes through layer norm as a **scaled sum** of per-term contributions
3. The scaling factor σ is shared, so relative contributions are preserved

## Implementation Steps

For target token t and expanded residual `x = Σᵢ xᵢ`:

1. **Get layer norm parameters**: `γ, β` from final layer norm
2. **Get unembedding vector**: `u = unembed[token_id]`
3. **For each term xᵢ**:
   - Compute mean: `μᵢ = mean(xᵢ)`
   - Compute raw contribution: `cᵢ = ⟨u ⊙ γ, xᵢ - μᵢ⟩`
4. **Normalize**: `contribution(xᵢ) = cᵢ / Σⱼ cⱼ`

The bias term `⟨u, β⟩` is constant across the expansion and can be reported separately.

## Experimental Results (pythia-160m-deduped)

### Finding: The Final Block Dominates

Testing with pythia-160m-deduped reveals that the **final block (Δx^11) contributes ~95% of the logit** for all predictions, regardless of whether they are context-dependent or not.

### Example 1: Context-Dependent Prediction

**Prompt**: "The capital of Ireland is"
**Target**: " Dublin" (ranked 4th, 2.67% probability)

```
T(" is") ▷ ' Dublin' {
  embed(' is'):   +0.1  (+0.0%)
  Δx^0:           -9.9  (-0.5%)
  Δx^1:           +0.3  (+0.0%)
  Δx^2:           +2.1  (+0.1%)
  Δx^3:           +3.3  (+0.2%)
  Δx^4:           -2.1  (-0.1%)
  Δx^5:           -0.5  (-0.0%)
  Δx^6:           +8.2  (+0.4%)
  Δx^7:           +5.0  (+0.3%)
  Δx^8:          +23.1  (+1.2%)
  Δx^9:           -9.2  (-0.5%)
  Δx^10:         +76.0  (+4.0%)
  Δx^11:       +1786.8  (+94.9%)   ← Final block dominates
  Total: 1883.3
}
```

### Example 2: Word Continuation

**Prompt**: "The un"
**Target**: "ic" (top prediction, 1.53% probability)

```
T(" un") ▷ 'ic' {
  embed(' un'):   negligible
  ...
  Δx^10:         +55.7  (+4.3%)
  Δx^11:       +1244.1  (+95.6%)   ← Same pattern
}
```

### Interpretation

The embedding's direct contribution to the logit is negligible because:

1. **Raw embeddings don't project to logit space** — The embedding vector isn't aligned with unembedding vectors
2. **The final block "translates" to output space** — Block 11 learns to produce vectors that project well onto unembedding directions
3. **Context integration happens earlier but manifests through final block** — For "Dublin", attention to "Ireland" in earlier blocks shapes the residual, but this information flows through to block 11 which produces the output

### What Block-Level Contribution Shows

Block-level contribution answers: **"Which block's output vector projects most onto the target unembedding?"**

It does NOT directly show:
- Where context information was integrated (that happens via attention in earlier blocks)
- The causal chain of how information flowed

### To See Context Integration

To understand *where* context matters, we need:
1. **Level 2 expansion** — Separate attention vs MLP within each block
2. **Attention pattern analysis** — Which heads in which layers attend to "Ireland"
3. **Ablation studies** — How does removing "Ireland" from context change block contributions?

## Open Questions

1. Should negative contributions be displayed differently? (A term can push away from a token)
2. How to handle very small total contributions (near-zero logit)?
3. Should we also report the raw contribution values, not just percentages?
4. Why does the final block dominate so strongly? Is this architecture-specific (GPT-NeoX)?
5. Would attention-head-level analysis (Level 2 expansion) reveal where context integration occurs?
