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

## Agent-Specific Guidelines

### Code Engineer Agent
- **Language**: Python with TransformerLens (Julia only if specific task strongly benefits)
- **Pattern**: Wrap computed vectors in types that track their symbolic origin as expressions
- Use TransformerLens hook points: `blocks.{i}.hook_resid_pre`, `hook_attn_out`, `hook_mlp_out`
- Model: Use Pythia models only (`pythia-70m-deduped`, `pythia-14m`) until project matures
- Comments should reference notation: `# x55 = x^5_5 (residual at block 5, position 5)`

### Tester Agent
- Validate decompositions match actual model outputs within tolerance
- Test pattern: compare `embed + Σ delta` against `hook_resid_post` (see `Annotate2.ipynb` cell patterns)
- Use `torch.isclose()` with appropriate `atol` for floating point comparisons
- Synthetic probes: test dot products like `<unembed_token, LN(residual)>` match logits

### Design Architect Agent
- Core types needed: `SymbolicVector`, `ResidualStream`, `HeadWrite`, `BlockContribution`
- Each type should carry: computed tensor + symbolic expression + provenance metadata
- Operations (`+`, `*`, `@`) should propagate expressions: `a + b` → `Expr(:+, a.expr, b.expr)`
- API should support: `expand(expr)` to show block decomposition, `simplify(expr)` to collapse terms

### Mathematical Reasoning Agent
- Reference: "A Mathematical Framework for Transformer Circuits" (Anthropic)
- Layer norm linearization: $\langle v, LN(x) \rangle \approx \sqrt{N} \cdot \cos\theta$ (see `Summary.ipynb`)
- Decomposition: $x^n = \underline{\text{tok}} + \sum_i \Delta x^i$ where $\Delta x^i = A^i + M^i$
- When proposing new decompositions, ground them in verifiable computations against actual model activations

### Documentation Agent
- Keep `doc/notation.md` synchronized with any notation changes in code or notebooks
- When notation evolves, update all three: code comments, notebook markdown cells, and `doc/notation.md`
- Document the rationale for notation choices (why this symbol, what alternatives were considered)
- Maintain a changelog of notation decisions to track evolution

## Development Workflow

```bash
# Python environment (TransformerLens)
pip install transformer_lens torch

# Run notebooks for experimentation
jupyter notebook notebooks/
```

## Key Files
- `README.md` - Project goals and tasks
- `doc/notation.md` - Notation specification (evolving)
- `notebooks/Summary.ipynb` - Mathematical framework and examples
- `notebooks/Annotate2.ipynb` - Working examples with TransformerLens

## Anti-Patterns to Avoid
- Don't use raw tensor indices without symbolic tracking
- Don't hardcode model dimensions—parameterize from model config
- Avoid notation changes without updating both docs and code
