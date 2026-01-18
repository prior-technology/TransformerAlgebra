# Object Model

This document describes the Python types in TransformerAlgebra and the operations that transform between them.

## Type Hierarchy

```
VectorLike (Protocol)
├── EmbeddingVector      embed(token)
├── ResidualVector       T(x) or T^i(x)
├── BlockContribution    ΔB^i
├── AttentionContribution ΔB^i_A
├── MLPContribution      ΔB^i_M
├── VectorSum            x + ΔB^1 + ΔB^2 + ...
└── LayerNormApplication LN^T(...)
```

All types implementing `VectorLike` have:
- `.tensor` → `torch.Tensor` of shape `[d_model]`
- `.d_model` → `int`

## Core Types

### PromptedTransformer

The entry point. Wraps a HuggingFace model with a fixed context.

```python
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
```

**Properties:**
- `T.config` → `ModelConfig` (n_layers, n_heads, d_model, vocab_size)
- `T.tokens` → `TokenInfo` (token_ids, tokens)

**Operations:**
- `T(token)` → `ResidualVector` — apply transformer to token
- `T.embed(token)` → `EmbeddingVector` — get raw embedding

### EmbeddingVector

The embedding of a token before any transformer blocks.

```python
x = T.embed(" is")  # embed(' is')
```

**Properties:**
- `.token_text`, `.token_id`
- `.tensor` → raw embedding vector

### ResidualVector

The result of applying the transformer (or truncated transformer) to an embedding.

```python
x = T(" is")       # T(embed(' is')) — full transformer
x = T(" is", 5)    # T^5(embed(' is')) — after 5 blocks
```

**Properties:**
- `.layer` — how many blocks have been applied
- `.position` — token position in sequence
- `.normed` → tensor with final layer norm applied

**Operations:**
- `.expand()` → `LayerNormApplication` — decompose into terms

### BlockContribution

The additive contribution from a single transformer block: ΔB^i = B^i(x) − x.

```python
ex = expand(T(" is"))
block_1 = ex.inner[1]  # ΔB^1
```

**Operations:**
- `.expand()` → `VectorSum[AttentionContribution, MLPContribution]`

### AttentionContribution / MLPContribution

The attention and MLP components of a block contribution.

```python
attn_mlp = expand(block_1)  # ΔB^1_A + ΔB^1_M
```

### VectorSum

An explicit sum of vector-like terms. Iterable and indexable.

```python
ex = expand(T(" is"))
inner = ex.inner        # VectorSum: embed(' is') + ΔB^1 + ... + ΔB^12
inner[0]                # EmbeddingVector
inner[5]                # BlockContribution for block 5
len(inner)              # number of terms
```

**Operations:**
- `.expand()` → `VectorSum` — expand each expandable term
- `[i]` → access individual terms
- `iter()` → iterate over terms

### LayerNormApplication

Wraps a vector expression with the terminal layer norm.

```python
ex = expand(T(" is"))  # LN^T(embed(' is') + ΔB^1 + ... + ΔB^12)
ex.inner               # the VectorSum inside
ex.tensor              # result after applying LN
```

**Properties:**
- `.inner` → the wrapped `VectorLike`
- `.unwrapped` → alias for `.inner`

**Operations:**
- `.expand()` → `LayerNormApplication` — expand inner, keep wrapper

## Transformation Graph

```
                    T.embed(token)
                         │
                         ▼
                  ┌─────────────┐
                  │EmbeddingVector│
                  └──────┬──────┘
                         │ T(x)
                         ▼
                  ┌─────────────┐
                  │ResidualVector│
                  └──────┬──────┘
                         │ expand()
                         ▼
              ┌──────────────────────┐
              │LayerNormApplication   │
              │ LN^T(VectorSum)       │
              └──────────┬───────────┘
                         │ .inner
                         ▼
              ┌──────────────────────┐
              │VectorSum              │
              │ embed + ΔB^1 + ...    │
              └──────────┬───────────┘
                         │ [i] (i > 0)
                         ▼
              ┌──────────────────────┐
              │BlockContribution      │
              │ ΔB^i                  │
              └──────────┬───────────┘
                         │ expand()
                         ▼
              ┌──────────────────────┐
              │VectorSum              │
              │ ΔB^i_A + ΔB^i_M      │
              └──────────────────────┘
```

## Analysis Functions

### logits(residual) → LogitMapping

Compute logits from a residual vector.

```python
L = logits(T(" is"))
L[" Dublin"]           # LogitValue: <unembed(' Dublin'), T(...)> = 12.34
L.top_k(5)             # top 5 predictions
```

### predict(residual) → ProbabilityMapping

Compute probabilities via softmax.

```python
P = predict(T(" is"))
P[" Dublin"]           # ProbabilityValue: P(' Dublin' | ...) = 45.2%
P.top_k(5)             # top 5 with probabilities
```

### expand(x) → VectorLike

Expand any expandable vector into its components.

```python
expand(T(" is"))       # → LayerNormApplication
expand(block)          # → VectorSum of attn + mlp
expand(embedding)      # → VectorSum([embedding])
```

### contribution(expanded, logit) → ContributionResult

Decompose how each term in an expansion contributes to a logit.

```python
x = T(" is")
ex = expand(x)
logit = logits(x)[" Dublin"]
c = contribution(ex.inner, logit)
c.summary()            # formatted breakdown
c.top_k(3)             # most impactful terms
```

## Usage Pattern

Typical analysis workflow:

```python
# 1. Create prompted transformer
T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

# 2. Run on next token
x = T(" is")

# 3. Check predictions
print(predict(x).summary())

# 4. Expand and analyze
ex = expand(x)
logit = logits(x)[" Dublin"]
contrib = contribution(ex.inner, logit)
print(contrib.summary())

# 5. Drill down into significant blocks
for term, value, pct in contrib.top_k(3):
    if hasattr(term, 'expand'):
        print(f"\n{term}:")
        sub = term.expand()
        print(f"  Attention + MLP: {sub}")
```
