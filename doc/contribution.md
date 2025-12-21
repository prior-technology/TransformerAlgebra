# Contribution Analysis

The `contribution` function shows how each term in an expanded expression contributes to the logit (and thus probability) of a target token.

## Notation

We define `▷` as the contribution operator. If `x = Σᵢ xᵢ` (an expanded residual):

```
x ▷ t = { x₀: 20%, x₁: 2%, x₂: 50%, ... }
```

reads as: "for token t, term x₀ contributes 20% of the logit, x₁ contributes 2%, etc."

## API

The contribution function acts on an expanded expression:

```python
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
x = T(" is")                           # Pre-LN residual
p = predict(x)[" Dublin"]              # ProbabilityValue for Dublin
contrib = contribution(expand(p))      # Contribution breakdown
print(contrib)
# T(x) ▷ ' Dublin' { embed(' is'): 12%, Δx^0: -2%, Δx^7: 42%, ... }
```

Or starting from logits:

```python
logit = logits(x)[" Dublin"]           # LogitValue for Dublin
contrib = contribution(expand(logit))  # Contribution breakdown
```

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

## Example 1: Context-Dependent Prediction

**Prompt**: "The capital of Ireland is"

For the token " Dublin", we expect the prediction to depend heavily on the context "Ireland", not just the immediate token " is". The contributions should show significant values from later blocks that have processed the attention to "Ireland".

```
T(" is") ▷ ' Dublin' {
  embed(' is'):  12%    # The word "is" alone doesn't predict Dublin
  Δx^0:          -2%    # Early blocks: minimal
  Δx^1:           3%
  ...
  Δx^7:          42%    # Later block: attention to "Ireland" → Dublin
  Δx^8:          28%
  ...
  Δx^{11}:        8%    # Final refinement
}
```

**Expected pattern**: Low embedding contribution, high contribution from blocks where attention integrates "Ireland" into the prediction.

## Example 2: Embedding-Dominated Prediction

**Prompt**: "un" (beginning of a multi-token word)

For tokens that continue the word (e.g., "likely", "til", "der"), the prediction should depend primarily on the embedding, not on context processing.

```
T("un") ▷ 'likely' {
  embed('un'):   85%    # The prefix strongly predicts continuations
  Δx^0:           3%
  Δx^1:           2%
  ...
  Δx^{11}:        1%
}
```

**Expected pattern**: High embedding contribution, low contribution from blocks (no informative context to integrate).

## Example 3: Comparing Tokens

The same residual can show different contribution patterns for different target tokens:

```
T(" is") ▷ ' Dublin' { embed: 12%, Δx^7: 42%, ... }
T(" is") ▷ ' the'    { embed: 45%, Δx^3: 20%, ... }
T(" is") ▷ ' a'      { embed: 52%, Δx^2: 15%, ... }
```

Common tokens like " the" and " a" may be predicted more by the embedding (they follow many words), while specific predictions like " Dublin" require context integration.

## Open Questions

1. Should negative contributions be displayed differently? (A term can push away from a token)
2. How to handle very small total contributions (near-zero logit)?
3. Should we also report the raw contribution values, not just percentages?
4. Shapley values for probability (not logit) contribution—when is this needed?
