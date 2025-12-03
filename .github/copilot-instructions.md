# TransformerAlgebra - AI Agent Instructions

## Project Vision

TransformerAlgebra provides a symbolic notation and interpreter for analyzing LLM internal states. The goal is to describe transformer operations as algebraic expressions where terms can be expanded, simplified, and interpreted—keeping close alignment between mathematical notation and executable code.

## Core Concepts

### Notation System (PLACEHOLDER - subject to revision)
Current working notation (see `doc/notation.md`, `notebooks/Summary.ipynb`):
- **Residual vectors**: $x^i_j$ = residual at block $i$, position $j$
- **Token embeddings**: $\underline{\text{token}}$ (underlined) = embedding vector for token
- **Unembeddings**: $\overline{\text{token}}$ (overlined) = unembedding vector
- **Block contributions**: $\Delta x^i_j$ = contribution from block $i$ at position $j$
- **Operations**: $A^{i,h}$ = attention pattern (layer $i$, head $h$), $M^i$ = MLP layer, $LN$ = layer normalization

**Notation is experimental**: Propose alternatives when current symbols are awkward. Document rationale for any changes.

### Key Abstractions (informed by prior Julia work)
```
Residual       - vector in transformer's residual space
PromptedTransformer - model + context, acts on residuals via `*` operator
Transformation - expression tracking origin of computed results
embed/unembed  - map tokens ↔ vectors, named by token text
```

## Key Files
- `README.md` - Project goals and tasks
- `doc/notation.md` - Notation specification (evolving)
- `notebooks/Summary.ipynb` - Mathematical framework and examples

