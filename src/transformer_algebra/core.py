"""Core classes for analyzing transformer internal states."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Protocol, runtime_checkable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from typing import Self


# =============================================================================
# LaTeX Rendering Utilities
# =============================================================================

def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in text for use in \\text{}.

    Handles common special characters that appear in token text.
    """
    # Order matters: escape backslash first
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('_', r'\_'),
        ('^', r'\^{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
    ]
    for char, escaped in replacements:
        text = text.replace(char, escaped)
    return text


def _latex_token(token_text: str) -> str:
    """Format a token for LaTeX display, escaping special characters."""
    return _latex_escape(token_text)


# =============================================================================
# VectorLike Protocol
# =============================================================================

@runtime_checkable
class VectorLike(Protocol):
    """Protocol for vector-like objects in expressions.

    All vector-valued objects (embeddings, residuals, contributions)
    implement this protocol, enabling uniform handling in expressions.
    """

    @property
    def tensor(self) -> torch.Tensor:
        """Evaluate to concrete tensor [d_model]."""
        ...

    @property
    def d_model(self) -> int:
        """Vector dimension."""
        ...


def load_pythia_model(model_name: str = "EleutherAI/pythia-160m-deduped"):
    """Load a Pythia model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        tuple: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return model, tokenizer


@dataclass
class ModelConfig:
    """Transformer model configuration."""
    name: str
    n_layers: int
    n_heads: int
    d_model: int
    vocab_size: int

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_huggingface(cls, model) -> "ModelConfig":
        """Extract config from a HuggingFace model."""
        hf_config = model.config
        return cls(
            name=hf_config.name_or_path,
            n_layers=hf_config.num_hidden_layers,
            n_heads=hf_config.num_attention_heads,
            d_model=hf_config.hidden_size,
            vocab_size=hf_config.vocab_size,
        )


@dataclass
class TokenInfo:
    """Information about tokenized text."""
    token_ids: list[int]
    tokens: list[str]

    def __repr__(self):
        return f"TokenInfo(tokens={self.tokens}, ids={self.token_ids})"


class EmbeddingVector:
    """An input embedding vector - the domain of T.

    Represents embed(token) - the token embedding before any transformer blocks.
    This is the input that T maps to T(embed(token)).

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, token_text: str, token_id: int,
                 transformer: "PromptedTransformer"):
        self._tensor = tensor
        self.token_text = token_text
        self.token_id = token_id
        self.transformer = transformer

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def d_model(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self):
        return f"embed({self.token_text!r})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        token = _latex_token(self.token_text)
        return rf"$\underline{{\text{{{token}}}}}$"


class ResidualVector:
    """A residual stream vector - the output of T acting on an embedding.

    Represents T^i(embed(token)) - the result after i transformer blocks.
    T is a nonlinear operator; T^n means the full transformer, T^5 means
    truncated after block 5.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, transformer: "PromptedTransformer",
                 layer: int, position: int, token_text: str,
                 hidden_states: tuple[torch.Tensor, ...] | None = None):
        self._tensor = tensor
        self.transformer = transformer
        self.layer = layer
        self.position = position
        self.token_text = token_text
        # Cache all hidden states for expand()
        self._hidden_states = hidden_states

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def d_model(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self):
        n_layers = self.transformer.config.n_layers
        embed_str = f"embed({self.token_text!r})"
        if self.layer == n_layers:
            return f"T({embed_str})"
        else:
            return f"T^{self.layer}({embed_str})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        token = _latex_token(self.token_text)
        embed_latex = rf"\underline{{\text{{{token}}}}}"
        n_layers = self.transformer.config.n_layers
        if self.layer == n_layers:
            return rf"$T({embed_latex})$"
        else:
            return rf"$T^{{{self.layer}}}({embed_latex})$"

    @property
    def normed(self) -> torch.Tensor:
        """Apply final layer norm to get the normalized residual."""
        return self.transformer._final_ln(self._tensor)

    def expand(self) -> "LayerNormApplication":
        """Expand into LN^T applied to embedding plus block contributions.

        T(x) = LN^T(x + ΔB^1(x) + ΔB^2(x^1) + ... + ΔB^n(x^{n-1}))

        Returns:
            LayerNormApplication wrapping VectorSum of embedding and block contributions
        """
        if self._hidden_states is None:
            raise ValueError(
                "Cannot expand: hidden states not cached. "
                "This ResidualVector was created without storing intermediate states."
            )

        terms: list[VectorLike] = []

        # First term: the embedding (hidden_states[0])
        embed_tensor = self._hidden_states[0][0, self.position, :]
        terms.append(EmbeddingVector(
            tensor=embed_tensor,
            token_text=self.token_text,
            token_id=self.transformer.get_token_id(self.token_text),
            transformer=self.transformer,
        ))

        # Block contributions: ΔB^i = hidden_states[i] - hidden_states[i-1]
        # Internal layer index is 0-based, display is 1-based per notation.md
        for i in range(self.layer):
            delta = (self._hidden_states[i + 1][0, self.position, :] -
                     self._hidden_states[i][0, self.position, :])
            terms.append(BlockContribution(
                tensor=delta,
                layer=i,  # 0-based internally
                transformer=self.transformer,
                token_text=self.token_text,
                position=self.position,
                hidden_states=self._hidden_states,
            ))

        inner_sum = VectorSum(terms)

        # Wrap with LN^T per notation.md: T(x) = LN^T(T^n(x))
        return LayerNormApplication(
            inner=inner_sum,
            layer_norm=self.transformer._final_ln,
            transformer=self.transformer,
            name="LN^T",
        )


class AttentionContribution:
    """Contribution from the attention sublayer of a block.

    Represents ΔB^i_A(x^{i-1}) - the attention component of block i's contribution.

    Note: Internal layer index is 0-based, display is 1-based per notation.md.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, layer: int,
                 transformer: "PromptedTransformer", token_text: str,
                 position: int):
        self._tensor = tensor
        self.layer = layer  # 0-based internally
        self.transformer = transformer
        self.token_text = token_text
        self.position = position

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def d_model(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self):
        # Display as operator composition per notation.md
        block_idx = self.layer + 1
        if self.layer == 0:
            return f"ΔB^{block_idx}_A"
        else:
            return f"ΔB^{block_idx}_A T^{self.layer}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        block_idx = self.layer + 1
        if self.layer == 0:
            return rf"$\Delta B^{{{block_idx}}}_A$"
        else:
            return rf"$\Delta B^{{{block_idx}}}_A T^{{{self.layer}}}$"


class MLPContribution:
    """Contribution from the MLP sublayer of a block.

    Represents ΔB^i_M(x^{i-1}) - the MLP component of block i's contribution.

    Note: Internal layer index is 0-based, display is 1-based per notation.md.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, layer: int,
                 transformer: "PromptedTransformer", token_text: str,
                 position: int):
        self._tensor = tensor
        self.layer = layer  # 0-based internally
        self.transformer = transformer
        self.token_text = token_text
        self.position = position

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def d_model(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self):
        # Display as operator composition per notation.md
        block_idx = self.layer + 1
        if self.layer == 0:
            return f"ΔB^{block_idx}_M"
        else:
            return f"ΔB^{block_idx}_M T^{self.layer}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        block_idx = self.layer + 1
        if self.layer == 0:
            return rf"$\Delta B^{{{block_idx}}}_M$"
        else:
            return rf"$\Delta B^{{{block_idx}}}_M T^{{{self.layer}}}$"


class BlockContribution:
    """Contribution from a single transformer block.

    Represents ΔB^i(x^{i-1}) = B^i(x^{i-1}) - x^{i-1}, the additive
    contribution from block i applied to the accumulated residual.
    Can be expanded into ΔB^i_A + ΔB^i_M.

    Note: Internal layer index is 0-based, display is 1-based per notation.md.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, layer: int,
                 transformer: "PromptedTransformer", token_text: str,
                 position: int,
                 hidden_states: tuple[torch.Tensor, ...] | None = None,
                 attention_tensor: torch.Tensor | None = None,
                 mlp_tensor: torch.Tensor | None = None):
        self._tensor = tensor
        self.layer = layer  # 0-based internally
        self.transformer = transformer
        self.token_text = token_text
        self.position = position
        # Store hidden states for computing sublayer contributions
        self._hidden_states = hidden_states
        # Cache attention/MLP components for expand()
        self._attention_tensor = attention_tensor
        self._mlp_tensor = mlp_tensor

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def d_model(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self):
        # Display as operator composition per notation.md:
        # ΔB^i T^{i-1} for i > 1, just ΔB^1 for i = 1 (since T^0 = I)
        block_idx = self.layer + 1  # 1-based display
        if self.layer == 0:
            return f"ΔB^{block_idx}"
        else:
            return f"ΔB^{block_idx} T^{self.layer}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        block_idx = self.layer + 1
        if self.layer == 0:
            return rf"$\Delta B^{{{block_idx}}}$"
        else:
            return rf"$\Delta B^{{{block_idx}}} T^{{{self.layer}}}$"

    def expand(self) -> "VectorSum":
        """Expand into attention + MLP contributions.

        ΔB^i T^{i-1} -> ΔB^i_A T^{i-1} + ΔB^i_M T^{i-1}

        Returns:
            VectorSum of attention and MLP contributions
        """
        if self._attention_tensor is None or self._mlp_tensor is None:
            if self._hidden_states is None:
                raise ValueError(
                    "Cannot expand BlockContribution: hidden states not available. "
                    "This BlockContribution was created without storing intermediate states."
                )
            # Compute attention and MLP contributions by running sublayers
            attn_tensor, mlp_tensor = self.transformer._compute_sublayer_contributions(
                self.layer, self.position, self._hidden_states
            )
            self._attention_tensor = attn_tensor
            self._mlp_tensor = mlp_tensor

        attn = AttentionContribution(
            tensor=self._attention_tensor,
            layer=self.layer,
            transformer=self.transformer,
            token_text=self.token_text,
            position=self.position,
        )
        mlp = MLPContribution(
            tensor=self._mlp_tensor,
            layer=self.layer,
            transformer=self.transformer,
            token_text=self.token_text,
            position=self.position,
        )
        return VectorSum([attn, mlp])


class VectorSum:
    """Sum of vector-like objects.

    Represents an expanded form like: (I + ΔB^1 + ΔB^2 T^1 + ... + ΔB^n T^{n-1})x

    Is itself VectorLike: can be used anywhere a vector is expected.
    """

    def __init__(self, terms: list[VectorLike]):
        if not terms:
            raise ValueError("VectorSum requires at least one term")
        self.terms = terms

    @property
    def tensor(self) -> torch.Tensor:
        """Sum of all term tensors."""
        result = self.terms[0].tensor.clone()
        for term in self.terms[1:]:
            result = result + term.tensor
        return result

    @property
    def d_model(self) -> int:
        return self.terms[0].d_model

    def __getitem__(self, i: int) -> VectorLike:
        return self.terms[i]

    def __len__(self) -> int:
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)

    def __repr__(self):
        # Display as factored form: (I + ΔB^1 + ΔB^2 T^1 + ...)x
        # when first term is an embedding
        if len(self.terms) > 1 and isinstance(self.terms[0], EmbeddingVector):
            embed = self.terms[0]
            # Operators: I for identity (embedding), then block contributions
            ops = ["I"] + [repr(t) for t in self.terms[1:]]
            return f"({' + '.join(ops)}){embed!r}"
        else:
            return " + ".join(repr(t) for t in self.terms)

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        def term_latex(term):
            """Get LaTeX for a term, stripping outer $...$."""
            if hasattr(term, '_repr_latex_'):
                latex = term._repr_latex_()
                # Strip outer $ signs for embedding in larger expression
                if latex.startswith('$') and latex.endswith('$'):
                    return latex[1:-1]
                return latex
            return repr(term)

        if len(self.terms) > 1 and isinstance(self.terms[0], EmbeddingVector):
            embed = self.terms[0]
            embed_latex = term_latex(embed)
            # Operators: I for identity, then block contributions
            ops = ["I"] + [term_latex(t) for t in self.terms[1:]]
            return rf"$({' + '.join(ops)}){embed_latex}$"
        else:
            terms_latex = [term_latex(t) for t in self.terms]
            return rf"${' + '.join(terms_latex)}$"

    def expand(self) -> "VectorSum":
        """Expand each term that can be expanded."""
        expanded: list[VectorLike] = []
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


# =============================================================================
# Expression Types for Symbolic Manipulation
# =============================================================================


class CenteredVector:
    """Mean-centered vector: P(x) = x - mean(x).

    This is the centering operation used inside layer normalization.
    Implements VectorLike protocol.
    """

    def __init__(self, inner: VectorLike):
        self.inner = inner

    @property
    def tensor(self) -> torch.Tensor:
        """Evaluate to centered tensor."""
        x = self.inner.tensor
        return x - x.mean()

    @property
    def d_model(self) -> int:
        return self.inner.d_model

    def __repr__(self):
        return f"P({self.inner!r})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        inner_latex = self.inner._repr_latex_() if hasattr(self.inner, '_repr_latex_') else repr(self.inner)
        if inner_latex.startswith('$') and inner_latex.endswith('$'):
            inner_latex = inner_latex[1:-1]
        return rf"$P({inner_latex})$"


class GammaScaled:
    """Gamma-scaled vector: γ ⊙ x (element-wise multiplication).

    This represents the learned scale applied in layer normalization.
    Implements VectorLike protocol.
    """

    def __init__(self, gamma: torch.Tensor, inner: VectorLike):
        self._gamma = gamma
        self.inner = inner

    @property
    def tensor(self) -> torch.Tensor:
        """Evaluate to gamma-scaled tensor."""
        return self._gamma * self.inner.tensor

    @property
    def d_model(self) -> int:
        return self.inner.d_model

    def __repr__(self):
        return f"γ⊙{self.inner!r}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        inner_latex = self.inner._repr_latex_() if hasattr(self.inner, '_repr_latex_') else repr(self.inner)
        if inner_latex.startswith('$') and inner_latex.endswith('$'):
            inner_latex = inner_latex[1:-1]
        return rf"$\gamma \odot {inner_latex}$"


class ScaledVector:
    """Scalar-scaled vector: c * x.

    Represents a vector multiplied by a scalar (like 1/σ in layer norm).
    Implements VectorLike protocol.
    """

    def __init__(self, scale: float, inner: VectorLike, scale_label: str = ""):
        self._scale = scale
        self.inner = inner
        self.scale_label = scale_label  # e.g., "1/σ" for display

    @property
    def scale(self) -> float:
        return self._scale

    @property
    def tensor(self) -> torch.Tensor:
        """Evaluate to scaled tensor."""
        return self._scale * self.inner.tensor

    @property
    def d_model(self) -> int:
        return self.inner.d_model

    def __repr__(self):
        label = self.scale_label if self.scale_label else f"{self._scale:.3f}"
        return f"{label}·{self.inner!r}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        inner_latex = self.inner._repr_latex_() if hasattr(self.inner, '_repr_latex_') else repr(self.inner)
        if inner_latex.startswith('$') and inner_latex.endswith('$'):
            inner_latex = inner_latex[1:-1]
        if self.scale_label:
            # Convert common labels to LaTeX
            label = self.scale_label.replace('σ', r'\sigma')
            return rf"${label} \cdot {inner_latex}$"
        else:
            return rf"${self._scale:.3f} \cdot {inner_latex}$"


class InnerProduct:
    """Inner product of two vector expressions: ⟨left, right⟩.

    First-class expression type that can be expanded through layer norms
    and vector sums.
    """

    def __init__(self, left: VectorLike, right: VectorLike, label: str = ""):
        self.left = left
        self.right = right
        self._label = label

    @property
    def value(self) -> float:
        """Evaluate the inner product to a scalar."""
        return torch.dot(self.left.tensor, self.right.tensor).item()

    def __repr__(self):
        if self._label:
            return f"⟨{self._label}, {self.right!r}⟩"
        return f"⟨{self.left!r}, {self.right!r}⟩"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        def get_latex(obj):
            if hasattr(obj, '_repr_latex_'):
                latex = obj._repr_latex_()
                if latex.startswith('$') and latex.endswith('$'):
                    return latex[1:-1]
                return latex
            return repr(obj)

        right_latex = get_latex(self.right)
        if self._label:
            # Labels like "γ⊙unembed(' token')" need conversion
            left_latex = self._label.replace('γ', r'\gamma').replace('⊙', r' \odot ')
            return rf"$\langle {left_latex}, {right_latex} \rangle$"
        left_latex = get_latex(self.left)
        return rf"$\langle {left_latex}, {right_latex} \rangle$"

    def __float__(self):
        return self.value

    def expand(self) -> "InnerProduct | ScalarSum":
        """Expand through layer norms and sums.

        ⟨u, LN(Σxᵢ)⟩ → (1/σ) * Σᵢ ⟨u⊙γ, P(xᵢ)⟩ + u·β
        ⟨u, Σxᵢ⟩ → Σᵢ ⟨u, xᵢ⟩
        """
        # Expand through LayerNormApplication
        if isinstance(self.right, LayerNormApplication):
            return self.right.expand_inner_product(self.left)

        # Expand through VectorSum
        if isinstance(self.right, VectorSum):
            terms = [InnerProduct(self.left, t) for t in self.right.terms]
            return ScalarSum(terms)

        return self


class ScalarSum:
    """Sum of scalar expressions.

    Represents Σᵢ sᵢ where each sᵢ is a scalar expression.
    """

    def __init__(self, terms: list["InnerProduct | ScalarValue"], bias: float = 0.0,
                 scale: float = 1.0, scale_label: str = ""):
        self.terms = terms
        self.bias = bias
        self._scale = scale
        self.scale_label = scale_label

    @property
    def value(self) -> float:
        """Evaluate the sum."""
        return self._scale * sum(float(t) for t in self.terms) + self.bias

    def __repr__(self):
        scale_str = self.scale_label if self.scale_label else (
            f"{self._scale:.3f}·" if abs(self._scale - 1.0) > 1e-6 else ""
        )
        terms_str = " + ".join(repr(t) for t in self.terms)
        if self.bias != 0:
            return f"{scale_str}({terms_str}) + {self.bias:.2f}"
        return f"{scale_str}({terms_str})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        def get_latex(obj):
            if hasattr(obj, '_repr_latex_'):
                latex = obj._repr_latex_()
                if latex.startswith('$') and latex.endswith('$'):
                    return latex[1:-1]
                return latex
            return repr(obj)

        terms_latex = [get_latex(t) for t in self.terms]
        terms_str = " + ".join(terms_latex)

        # Convert scale label to LaTeX (e.g., "1/σ" -> "1/\sigma")
        if self.scale_label:
            scale_str = self.scale_label.replace('σ', r'\sigma')
        elif abs(self._scale - 1.0) > 1e-6:
            scale_str = f"{self._scale:.3f}"
        else:
            scale_str = ""

        if scale_str:
            if self.bias != 0:
                return rf"${scale_str} \cdot ({terms_str}) + {self.bias:.2f}$"
            return rf"${scale_str} \cdot ({terms_str})$"
        else:
            if self.bias != 0:
                return rf"$({terms_str}) + {self.bias:.2f}$"
            return rf"${terms_str}$"

    def __float__(self):
        return self.value

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms)


class ScalarValue:
    """A labeled scalar value."""

    def __init__(self, value: float, label: str = ""):
        self._value = value
        self.label = label

    @property
    def value(self) -> float:
        return self._value

    def __float__(self):
        return self._value

    def __repr__(self):
        if self.label:
            return f"{self.label}={self._value:.2f}"
        return f"{self._value:.2f}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        if self.label:
            # Convert Greek letters
            label = self.label.replace('σ', r'\sigma').replace('β', r'\beta')
            return rf"${label} = {self._value:.2f}$"
        return rf"${self._value:.2f}$"


class LayerNormApplication:
    """Application of layer norm to a vector expression.

    Represents LN^T(inner) where inner is typically a VectorSum.

    Per notation.md:
        T(x) = LN^T(T^n(x)) = LN^T(x + Σ ΔB^i(x^{i-1}))

    This wrapper makes the layer norm explicit in expanded forms.
    """

    def __init__(self, inner: VectorLike, layer_norm, transformer: "PromptedTransformer",
                 name: str = "LN^T"):
        self.inner = inner
        self._layer_norm = layer_norm
        self.transformer = transformer
        self.name = name

    @property
    def tensor(self) -> torch.Tensor:
        """Apply layer norm to get the normalized tensor."""
        return self._layer_norm(self.inner.tensor)

    @property
    def d_model(self) -> int:
        return self.inner.d_model

    def __repr__(self):
        return f"{self.name}({self.inner})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        inner_latex = self.inner._repr_latex_() if hasattr(self.inner, '_repr_latex_') else repr(self.inner)
        if inner_latex.startswith('$') and inner_latex.endswith('$'):
            inner_latex = inner_latex[1:-1]
        # Convert name like "LN^T" to LaTeX "\mathrm{LN}^T"
        name_latex = self.name.replace('LN', r'\mathrm{LN}')
        return rf"${name_latex}({inner_latex})$"

    def expand(self) -> "LayerNormApplication":
        """Expand the inner expression while keeping LN wrapper."""
        if hasattr(self.inner, 'expand'):
            expanded_inner = self.inner.expand()
            return LayerNormApplication(
                inner=expanded_inner,
                layer_norm=self._layer_norm,
                transformer=self.transformer,
                name=self.name,
            )
        return self

    @property
    def unwrapped(self) -> VectorLike:
        """Get the inner expression without the LN wrapper."""
        return self.inner

    @property
    def gamma(self) -> torch.Tensor:
        """Layer norm scale parameter γ."""
        return self._layer_norm.weight.detach()

    @property
    def beta(self) -> torch.Tensor:
        """Layer norm shift parameter β."""
        return self._layer_norm.bias.detach()

    @property
    def eps(self) -> float:
        """Layer norm epsilon."""
        return self._layer_norm.eps

    def sigma(self) -> float:
        """Compute the standard deviation of the total (centered) residual.

        σ = ||P(x)||/√N where P(x) = x - mean(x)
        """
        total = self.inner.tensor
        centered = total - total.mean()
        var = (centered ** 2).mean()
        return torch.sqrt(var + self.eps).item()

    def expand_inner_product(self, vector: VectorLike) -> "ScalarSum":
        """Expand an inner product ⟨vector, LN(inner)⟩ through the layer norm.

        ⟨u, LN(Σxᵢ)⟩ = (1/σ) * Σᵢ ⟨u⊙γ, P(xᵢ)⟩ + u·β

        where:
        - σ is the standard deviation of the total residual
        - P(xᵢ) = xᵢ - mean(xᵢ) is the centered term
        - u⊙γ is element-wise product of vector with gamma

        Args:
            vector: The left vector in the inner product (e.g., unembedding)

        Returns:
            ScalarSum with individual term contributions
        """
        # Expand inner to VectorSum if needed
        inner = self.inner
        if hasattr(inner, 'expand') and not isinstance(inner, VectorSum):
            inner = inner.expand()

        # Get terms (either from VectorSum or wrap single term)
        if isinstance(inner, VectorSum):
            terms = inner.terms
        else:
            terms = [inner]

        # Compute sigma from total
        sigma = self.sigma()

        # Compute bias term: u · β
        u = vector.tensor
        bias_term = torch.dot(u, self.beta).item()

        # Create scaled inner products for each term
        # Each term is ⟨u⊙γ, P(xᵢ)⟩
        u_gamma = GammaScaled(self.gamma, vector)
        inner_products = []
        for term in terms:
            centered = CenteredVector(term)
            ip = InnerProduct(u_gamma, centered)
            inner_products.append(ip)

        return ScalarSum(
            terms=inner_products,
            bias=bias_term,
            scale=1.0 / sigma,
            scale_label="1/σ",
        )


def expand(x: VectorLike) -> VectorLike:
    """Expand a vector into its additive components.

    For ResidualVector: T(x) -> LN^T(x + ΔB^1(x) + ΔB^2(x^1) + ... + ΔB^n(x^{n-1}))
    For LayerNormApplication: expand the inner expression, keep LN wrapper
    For BlockContribution: ΔB^i -> ΔB^i_A + ΔB^i_M
    For VectorSum: expand each term recursively
    For other VectorLike: wrap in single-term VectorSum

    Args:
        x: Any vector-like object

    Returns:
        Expanded form (LayerNormApplication, VectorSum, or wrapped VectorLike)

    Example:
        >>> x = T(" is")
        >>> ex = expand(x)
        >>> print(ex)  # LN^T(embed(' is') + ΔB^1 + ΔB^2 + ... + ΔB^{12})
        >>> print(ex.inner)  # embed(' is') + ΔB^1 + ... (VectorSum)
    """
    if hasattr(x, 'expand'):
        return x.expand()
    return VectorSum([x])


class LogitMapping:
    """A mapping from tokens to logits with symbolic representation.

    Supports subscripting: logits["Dublin"] returns the logit for Dublin.
    """

    def __init__(self, logits_tensor: torch.Tensor, residual: ResidualVector):
        self._logits = logits_tensor
        self._residual = residual
        self._transformer = residual.transformer

    def __repr__(self):
        return f"logits({self._residual})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        residual_latex = self._residual._repr_latex_() if hasattr(self._residual, '_repr_latex_') else repr(self._residual)
        if residual_latex.startswith('$') and residual_latex.endswith('$'):
            residual_latex = residual_latex[1:-1]
        return rf"$\mathrm{{logits}}({residual_latex})$"

    def __getitem__(self, token_text: str) -> "LogitValue":
        """Get the logit for a specific token."""
        token_id = self._transformer.get_token_id(token_text)
        value = self._logits[token_id].item()
        return LogitValue(value, token_text, self._residual)

    def top_k(self, k: int = 5) -> list[tuple[str, float]]:
        """Get the top-k tokens by logit value."""
        values, indices = torch.topk(self._logits, k)
        result = []
        for v, idx in zip(values.tolist(), indices.tolist()):
            token = self._transformer.tokenizer.decode([idx])
            result.append((token, v))
        return result

    def summary(self, k: int = 5) -> str:
        """Return a formatted summary of top predictions."""
        lines = [f"Top {k} predictions:"]
        for i, (token, logit) in enumerate(self.top_k(k), 1):
            lines.append(f"  {i}. {token!r} (logit: {logit:.2f})")
        return "\n".join(lines)


class LogitValue:
    """A single logit value with symbolic representation.

    Represents <unembed(token), T(x)> - the inner product of the
    unembedding vector with the transformer output.
    """

    def __init__(self, value: float, token_text: str, residual: ResidualVector):
        self.value = value
        self.token_text = token_text
        self._residual = residual

    def __repr__(self):
        return f"<unembed({self.token_text!r}), {self._residual}> = {self.value:.2f}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        token = _latex_token(self.token_text)
        unembed_latex = rf"\overline{{\text{{{token}}}}}"
        residual_latex = self._residual._repr_latex_() if hasattr(self._residual, '_repr_latex_') else repr(self._residual)
        if residual_latex.startswith('$') and residual_latex.endswith('$'):
            residual_latex = residual_latex[1:-1]
        return rf"$\langle {unembed_latex}, {residual_latex} \rangle = {self.value:.2f}$"

    def __float__(self):
        return self.value


def logits(residual: ResidualVector) -> LogitMapping:
    """Compute logits from a residual vector.

    Args:
        residual: A ResidualVector from T(token)

    Returns:
        LogitMapping that can be subscripted by token
    """
    normed = residual.normed
    logits_tensor = residual.transformer._unembed(normed)
    return LogitMapping(logits_tensor, residual)


class ProbabilityMapping:
    """A mapping from tokens to probabilities with symbolic representation.

    Supports subscripting: predict(x)["Dublin"] returns the probability for Dublin.
    """

    def __init__(self, probs_tensor: torch.Tensor, logits_tensor: torch.Tensor,
                 residual: ResidualVector):
        self._probs = probs_tensor
        self._logits = logits_tensor
        self._residual = residual
        self._transformer = residual.transformer

    def __repr__(self):
        return f"P(token | {self._residual})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        residual_latex = self._residual._repr_latex_() if hasattr(self._residual, '_repr_latex_') else repr(self._residual)
        if residual_latex.startswith('$') and residual_latex.endswith('$'):
            residual_latex = residual_latex[1:-1]
        return rf"$P(\mathrm{{token}} \mid {residual_latex})$"

    def __getitem__(self, token_text: str) -> "ProbabilityValue":
        """Get the probability for a specific token."""
        token_id = self._transformer.get_token_id(token_text)
        prob = self._probs[token_id].item()
        logit = self._logits[token_id].item()
        return ProbabilityValue(prob, logit, token_text, self._residual)

    def top_k(self, k: int = 5) -> list[tuple[str, float, float]]:
        """Get the top-k tokens by probability.

        Returns:
            List of (token, probability, logit) tuples
        """
        values, indices = torch.topk(self._probs, k)
        result = []
        for p, idx in zip(values.tolist(), indices.tolist()):
            token = self._transformer.tokenizer.decode([idx])
            logit = self._logits[idx].item()
            result.append((token, p, logit))
        return result

    def summary(self, k: int = 5) -> str:
        """Return a formatted summary of top predictions."""
        lines = [f"Top {k} predictions:"]
        for i, (token, prob, logit) in enumerate(self.top_k(k), 1):
            lines.append(f"  {i}. {token!r} ({prob:.2%}, logit={logit:.2f})")
        return "\n".join(lines)

    @property
    def log_partition(self) -> float:
        """The log partition function log(Z) for this distribution."""
        return torch.logsumexp(self._logits, dim=0).item()


class ProbabilityValue:
    """A single probability value with symbolic representation.

    Represents P(token | x) = softmax(<unembed(token), T(x)>).
    """

    def __init__(self, prob: float, logit: float, token_text: str,
                 residual: ResidualVector):
        self.prob = prob
        self.logit = logit
        self.token_text = token_text
        self._residual = residual

    def __repr__(self):
        return f"P({self.token_text!r} | {self._residual}) = {self.prob:.2%}"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        token = _latex_token(self.token_text)
        residual_latex = self._residual._repr_latex_() if hasattr(self._residual, '_repr_latex_') else repr(self._residual)
        if residual_latex.startswith('$') and residual_latex.endswith('$'):
            residual_latex = residual_latex[1:-1]
        return rf"$P(\text{{{token}}} \mid {residual_latex}) = {self.prob:.2%}$"

    def __float__(self):
        return self.prob

    @property
    def log_prob(self) -> float:
        """Log probability (more numerically stable for small values)."""
        return float(torch.log(torch.tensor(self.prob)).item())


def predict(residual: ResidualVector) -> ProbabilityMapping:
    """Compute next-token probabilities from a residual vector.

    Applies softmax to logits to get a probability distribution over tokens.

    Args:
        residual: A ResidualVector from T(token)

    Returns:
        ProbabilityMapping that can be subscripted by token
    """
    normed = residual.normed
    logits_tensor = residual.transformer._unembed(normed)
    probs_tensor = torch.softmax(logits_tensor, dim=0)
    return ProbabilityMapping(probs_tensor, logits_tensor, residual)


class PromptedTransformer:
    """A transformer model with a specific prompt/context.

    Corresponds to T(context) in the notation - a model bound to a context
    that caches all intermediate states.

    Example:
        >>> model, tokenizer = load_pythia_model()
        >>> T = PromptedTransformer(model, tokenizer, "The capital of Ireland is")
        >>> print(T.config)
        >>> print(T.tokens)
    """

    def __init__(self, model, tokenizer, prompt: str):
        """Initialize with a model, tokenizer, and prompt.

        Args:
            model: A HuggingFace causal LM (GPT-NeoX architecture)
            tokenizer: The corresponding tokenizer
            prompt: The context/prompt text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.config = ModelConfig.from_huggingface(model)

        # Architecture-specific component access (GPT-NeoX)
        self._final_ln = model.gpt_neox.final_layer_norm
        self._unembed = model.embed_out

        # Tokenize and cache
        self._token_info = self._tokenize(prompt)

        # Run model and cache hidden states
        self._hidden_states, self._outputs = self._run_model(prompt)

    @property
    def tokens(self) -> TokenInfo:
        """Token information for the prompt."""
        return self._token_info

    @property
    def n_positions(self) -> int:
        """Number of token positions in the prompt."""
        return len(self._token_info.token_ids)

    def __repr__(self):
        n = len(self._token_info.tokens)
        return f"T(<{n} tokens>)"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        n = len(self._token_info.tokens)
        return rf"$T(\langle {n} \text{{ tokens}} \rangle)$"

    def embed(self, token_text: str) -> EmbeddingVector:
        """Get the embedding vector for a token.

        Args:
            token_text: Token text (e.g., " is")

        Returns:
            EmbeddingVector representing embed(token)

        Example:
            >>> T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
            >>> x = T.embed(" is")  # embedding vector for " is"
            >>> T(x)                # apply transformer to embedding
        """
        token_id = self.get_token_id(token_text)
        # Get embedding matrix and extract row for this token
        embed_matrix = self.model.gpt_neox.embed_in.weight
        tensor = embed_matrix[token_id, :].detach()
        return EmbeddingVector(
            tensor=tensor,
            token_text=token_text,
            token_id=token_id,
            transformer=self,
        )

    def __call__(self, token: str | EmbeddingVector, layer: int = -1) -> ResidualVector:
        """Apply the transformer to a token or embedding vector.

        T is a nonlinear operator mapping embed(token) → T(embed(token)).
        T^i means running the first i blocks (without final layer norm).

        Args:
            token: Either a token string (e.g., " is") or an EmbeddingVector
            layer: Which layer's residual to return (-1 = final, default)

        Returns:
            ResidualVector representing T^layer(embed(token)) - the pre-LN residual

        Example:
            >>> T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
            >>> T(" is")           # shorthand - returns pre-LN residual
            >>> T(" is").normed    # apply final layer norm
        """
        # Handle both string and EmbeddingVector inputs
        if isinstance(token, EmbeddingVector):
            token_text = token.token_text
        else:
            token_text = token

        # Extend prompt with new token and run model
        full_prompt = self.prompt + token_text
        extended_tokens = self.tokenizer(full_prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**extended_tokens, output_hidden_states=True)

        # Get residual at specified layer
        if layer == -1:
            layer = self.config.n_layers

        # HuggingFace hidden_states[n_layers] is POST final_layer_norm, but we want
        # pre-LN residuals for consistent block contribution semantics.
        # Compute the true pre-LN residual for the final layer.
        hidden_states = self._compute_pre_ln_hidden_states(
            outputs.hidden_states, extended_tokens
        )

        residual_tensor = hidden_states[layer][0, -1, :]

        return ResidualVector(
            tensor=residual_tensor,
            transformer=self,
            layer=layer,
            position=-1,
            token_text=token_text,
            hidden_states=hidden_states,  # Cache for expand()
        )

    def _tokenize(self, text: str) -> TokenInfo:
        """Tokenize text and return token information."""
        tokens = self.tokenizer(text, return_tensors="pt")
        token_ids = tokens["input_ids"][0].tolist()
        str_tokens = [self.tokenizer.decode([t]) for t in token_ids]
        return TokenInfo(token_ids=token_ids, tokens=str_tokens)

    def _run_model(self, prompt: str):
        """Run the model and return hidden states at all layers."""
        tokens = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
        return outputs.hidden_states, outputs

    def _compute_pre_ln_hidden_states(
        self,
        hf_hidden_states: tuple[torch.Tensor, ...],
        tokens: dict,
    ) -> tuple[torch.Tensor, ...]:
        """Compute hidden states with pre-LN final residual.

        HuggingFace GPT-NeoX returns hidden_states[n_layers] as POST final_layer_norm,
        but for consistent block contribution semantics, we need the PRE-LN residual.

        This method computes the true pre-LN final residual by running the last
        block manually on hidden_states[n_layers-1].

        Args:
            hf_hidden_states: Hidden states from HuggingFace model
            tokens: Tokenized input for attention mask

        Returns:
            Tuple of hidden states where all entries are pre-LN residuals
        """
        n_layers = self.config.n_layers

        # hidden_states[0..n_layers-1] are already pre-LN (raw residuals)
        # hidden_states[n_layers] is post-LN, we need to recompute it

        # Get the residual before the last block
        h_before_last = hf_hidden_states[n_layers - 1]

        # Run the last block to get the pre-LN output
        last_block = self.model.gpt_neox.layers[n_layers - 1]

        with torch.no_grad():
            # GPT-NeoX parallel architecture
            attn_ln_out = last_block.input_layernorm(h_before_last)
            mlp_ln_out = last_block.post_attention_layernorm(h_before_last)

            # Compute position embeddings for attention
            seq_len = h_before_last.shape[1]
            attention_mask = tokens.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones(1, seq_len, device=h_before_last.device)
            attention_mask = attention_mask.float()
            position_ids = torch.arange(seq_len, device=h_before_last.device).unsqueeze(0)
            position_embeddings = self.model.gpt_neox.rotary_emb(attn_ln_out, position_ids)

            # Run attention
            attn_output = last_block.attention(
                attn_ln_out,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )[0]

            # Run MLP
            mlp_output = last_block.mlp(mlp_ln_out)

            # Parallel residual: h_new = h + attn + mlp
            h_pre_ln_final = h_before_last + attn_output + mlp_output

        # Build corrected hidden states tuple
        # Keep hidden_states[0..n_layers-1] as-is, replace hidden_states[n_layers]
        corrected = list(hf_hidden_states[:n_layers]) + [h_pre_ln_final]
        return tuple(corrected)

    def get_token_id(self, token_text: str) -> int:
        """Get the token ID for a piece of text (should be a single token).

        Args:
            token_text: Text to convert (e.g., " Dublin" with leading space)

        Returns:
            Token ID

        Raises:
            ValueError: If the text tokenizes to multiple tokens
        """
        token_ids = self.tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                f"'{token_text}' tokenizes to {len(token_ids)} tokens, expected 1. "
                f"Token IDs: {token_ids}"
            )
        return token_ids[0]

    def residual(self, layer: int, position: int = -1) -> torch.Tensor:
        """Get the residual vector after a specific layer at a position.

        Args:
            layer: Layer index (0 = after embedding, 1-n_layers = after each block)
            position: Token position (default: last)

        Returns:
            Residual vector [d_model]
        """
        return self._hidden_states[layer][0, position, :]

    def residual_normed(self, layer: int, position: int = -1) -> torch.Tensor:
        """Get the layer-normalized residual vector.

        Args:
            layer: Layer index
            position: Token position (default: last)

        Returns:
            Normalized residual vector [d_model]
        """
        return self._final_ln(self.residual(layer, position))

    def logits_at_layer(self, layer: int, position: int = -1) -> torch.Tensor:
        """Compute logits from residual at a specific layer/position.

        This applies the final layer norm and unembedding to get
        what the model would predict if this were the final layer.

        Args:
            layer: Layer index
            position: Token position (default: last)

        Returns:
            Logits tensor [vocab_size]
        """
        resid_normed = self.residual_normed(layer, position)
        return self._unembed(resid_normed)

    def _compute_sublayer_contributions(
        self, layer: int, position: int,
        hidden_states: tuple[torch.Tensor, ...] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention and MLP contributions for a block.

        For GPT-NeoX (Pythia), the architecture uses parallel attention and MLP:
            x_new = x + attention(ln(x)) + mlp(ln(x))

        This method runs each sublayer separately to extract individual contributions.

        Args:
            layer: Block index (0 to n_layers-1)
            position: Token position
            hidden_states: Optional tuple of hidden states to use (if None, uses cached)

        Returns:
            Tuple of (attention_contribution, mlp_contribution) tensors
        """
        # Use provided hidden states or fall back to cached ones
        states = hidden_states if hidden_states is not None else self._hidden_states

        # Get the residual before this block
        residual_before = states[layer][0:1, :, :]  # Keep batch dim

        # Get the transformer block
        block = self.model.gpt_neox.layers[layer]

        with torch.no_grad():
            # GPT-NeoX parallel architecture:
            # - attention uses input_layernorm
            # - MLP uses post_attention_layernorm (in parallel, not after attention)
            attn_ln_out = block.input_layernorm(residual_before)
            mlp_ln_out = block.post_attention_layernorm(residual_before)

            # Compute attention contribution
            # GPT-NeoX attention uses rotary position embeddings
            seq_len = residual_before.shape[1]
            attention_mask = torch.ones(1, seq_len, device=residual_before.device)

            # Compute rotary position embeddings
            position_ids = torch.arange(seq_len, device=residual_before.device).unsqueeze(0)
            position_embeddings = self.model.gpt_neox.rotary_emb(attn_ln_out, position_ids)

            attn_output = block.attention(
                attn_ln_out,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            # attn_output is a tuple: (hidden_states, present_key_value)
            attn_contribution = attn_output[0][0, position, :]

            # Compute MLP contribution using post_attention_layernorm
            mlp_contribution = block.mlp(mlp_ln_out)[0, position, :]

        return attn_contribution, mlp_contribution


# =============================================================================
# Contribution Analysis
# =============================================================================


class UnembeddingVector:
    """An unembedding vector for a token.

    Represents the row of W_U corresponding to a token - used for computing logits.
    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, token_text: str, token_id: int,
                 transformer: "PromptedTransformer"):
        self._tensor = tensor
        self.token_text = token_text
        self.token_id = token_id
        self.transformer = transformer

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def d_model(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self):
        return f"unembed({self.token_text!r})"

    def _repr_latex_(self):
        """LaTeX representation for Jupyter notebooks."""
        token = _latex_token(self.token_text)
        return rf"$\overline{{\text{{{token}}}}}$"


class ContributionResult:
    """Result of contribution analysis showing how each term affects an inner product.

    Displays how terms in an expanded vector sum contribute to an inner product
    like a logit ⟨unembed(token), LN(x)⟩.

    The logit decomposes as:
        z = (1/σ) * Σᵢ ⟨u⊙γ, P(xᵢ)⟩ + b_t

    where σ is the std of the total residual and b_t = u·β is the bias term.
    """

    def __init__(
        self,
        vector: VectorLike,
        terms: list[VectorLike],
        raw_contributions: list[float],
        normalized_contributions: list[float],
        percentages: list[float],
        total_raw: float,
        sigma: float,
        beta_term: float,
        label: str = "",
    ):
        self.vector = vector  # The vector we're taking inner product with
        self.terms = terms
        self.raw_contributions = raw_contributions  # ⟨u⊙γ, P(xᵢ)⟩ before dividing by σ
        self.normalized_contributions = normalized_contributions  # After dividing by σ
        self.percentages = percentages
        self.total_raw = total_raw
        self.sigma = sigma  # Standard deviation of total residual
        self.beta_term = beta_term  # b_t = u·β
        self._label = label

    @property
    def target_token(self) -> str:
        """The target token (for unembedding vectors)."""
        if isinstance(self.vector, UnembeddingVector):
            return self.vector.token_text
        return self._label or repr(self.vector)

    @property
    def computed_value(self) -> float:
        """The inner product computed from contributions: sum(normalized) + beta_term."""
        return sum(self.normalized_contributions) + self.beta_term

    # Backwards compatibility alias
    @property
    def computed_logit(self) -> float:
        """Alias for computed_value (for backwards compatibility)."""
        return self.computed_value

    def __repr__(self):
        lines = [f"Contributions to ⟨{self.target_token}, LN(...)⟩:"]
        for term, norm, pct in zip(self.terms, self.normalized_contributions, self.percentages):
            sign = "+" if norm >= 0 else ""
            lines.append(f"  {repr(term)}: {sign}{norm:.2f} ({pct:+.1%})")
        lines.append(f"  Subtotal: {sum(self.normalized_contributions):.2f}")
        lines.append(f"  Beta term: {self.beta_term:.2f}")
        lines.append(f"  Total: {self.computed_value:.2f}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        """Iterate over (term, normalized_contribution, percentage) tuples."""
        return zip(self.terms, self.normalized_contributions, self.percentages)

    def top_k(self, k: int = 5) -> list[tuple[VectorLike, float, float]]:
        """Return the k terms with largest absolute normalized contribution."""
        items = list(zip(self.terms, self.normalized_contributions, self.percentages))
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return items[:k]

    def summary(self, k: int = 5) -> str:
        """Return a formatted summary of top contributions."""
        lines = [f"Top {k} contributions to {self.target_token!r}:"]
        for term, norm, pct in self.top_k(k):
            sign = "+" if norm >= 0 else ""
            lines.append(f"  {repr(term)}: {sign}{norm:.2f} ({pct:+.1%})")
        lines.append(f"  (σ={self.sigma:.2f}, β_t={self.beta_term:.2f})")
        return "\n".join(lines)


def contribution(
    expr: "LayerNormApplication | VectorSum",
    vector: "VectorLike | LogitValue | str",
    transformer: "PromptedTransformer | None" = None,
) -> ContributionResult:
    """Compute how each term in an expression contributes to an inner product.

    Given a LayerNormApplication or VectorSum and a vector, computes each term's
    contribution to the inner product ⟨vector, expr⟩.

    The inner product ⟨u, LN(Σᵢ xᵢ)⟩ decomposes as:
        value = (1/σ) * Σᵢ cᵢ + bias
    where cᵢ = ⟨u ⊙ γ, xᵢ - μᵢ⟩

    See doc/contribution.md and doc/analysis/speculation.md for mathematical details.

    Args:
        expr: A LayerNormApplication or VectorSum from expand(residual)
        vector: The vector to take inner product with. Can be:
            - VectorLike (any vector)
            - LogitValue (extracts unembedding vector)
            - str (token text, requires transformer)
        transformer: Required if vector is a string

    Returns:
        ContributionResult with breakdown by term

    Example:
        >>> T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
        >>> x = T(" is")
        >>> ex = expand(x)  # Returns LayerNormApplication
        >>> contrib = contribution(ex, " Dublin", T)  # Use token string
        >>> print(contrib.summary())

        # Or with LogitValue for backwards compatibility:
        >>> logit = logits(x)[" Dublin"]
        >>> contrib = contribution(ex.inner, logit)
    """
    # Handle LayerNormApplication - extract VectorSum and layer norm params
    if isinstance(expr, LayerNormApplication):
        ln_app = expr
        # Expand inner if needed
        inner = ln_app.inner
        if hasattr(inner, 'expand') and not isinstance(inner, VectorSum):
            inner = inner.expand()
        if isinstance(inner, VectorSum):
            terms = list(inner.terms)
        else:
            terms = [inner]

        gamma = ln_app.gamma
        beta = ln_app.beta
        eps = ln_app.eps
        xformer = ln_app.transformer

    elif isinstance(expr, VectorSum):
        # Legacy path: VectorSum requires inferring transformer from vector
        terms = list(expr.terms)
        gamma = None
        beta = None
        eps = 1e-5
        xformer = None

    else:
        raise TypeError(
            f"Expected LayerNormApplication or VectorSum, got {type(expr).__name__}"
        )

    # Resolve the vector argument
    label = ""
    if isinstance(vector, LogitValue):
        # Extract unembedding vector from LogitValue
        xformer = vector._residual.transformer
        token_id = xformer.get_token_id(vector.token_text)
        unembed_matrix = xformer._unembed.weight
        u_tensor = unembed_matrix[token_id, :].detach()
        u = UnembeddingVector(u_tensor, vector.token_text, token_id, xformer)
        label = vector.token_text

        # Get layer norm params if not already set
        if gamma is None:
            final_ln = xformer._final_ln
            gamma = final_ln.weight.detach()
            beta = final_ln.bias.detach()
            eps = final_ln.eps

    elif isinstance(vector, str):
        # Token string - need transformer
        if transformer is None and xformer is None:
            raise ValueError("transformer required when vector is a token string")
        xformer = transformer or xformer
        token_id = xformer.get_token_id(vector)
        unembed_matrix = xformer._unembed.weight
        u_tensor = unembed_matrix[token_id, :].detach()
        u = UnembeddingVector(u_tensor, vector, token_id, xformer)
        label = vector

        # Get layer norm params if not already set
        if gamma is None:
            final_ln = xformer._final_ln
            gamma = final_ln.weight.detach()
            beta = final_ln.bias.detach()
            eps = final_ln.eps

    else:
        # Assume VectorLike
        u = vector
        if gamma is None:
            raise ValueError(
                "LayerNormApplication required when using raw VectorLike, "
                "or pass a LogitValue/token string to infer layer norm params"
            )

    if len(terms) == 0:
        raise ValueError("Expression has no terms")

    # Compute beta term: b_t = u · β
    beta_term = torch.dot(u.tensor, beta).item()

    # Precompute u ⊙ γ (element-wise product)
    u_gamma = u.tensor * gamma  # [d_model]

    # Compute the total residual to get sigma (standard deviation)
    total_residual = sum(t.tensor for t in terms)
    centered_total = total_residual - total_residual.mean()
    var_total = (centered_total ** 2).mean()
    sigma = torch.sqrt(var_total + eps).item()

    # Compute contribution for each term
    raw_contributions = []
    for term in terms:
        x_i = term.tensor.detach()  # [d_model]
        mu_i = x_i.mean()  # scalar
        centered = x_i - mu_i  # [d_model]
        c_i = torch.dot(u_gamma, centered).item()  # scalar
        raw_contributions.append(c_i)

    # Normalize contributions by sigma
    normalized_contributions = [c / sigma for c in raw_contributions]

    # Compute total and percentages (based on normalized contributions)
    total_normalized = sum(normalized_contributions)
    total_raw = sum(raw_contributions)
    if abs(total_normalized) < 1e-10:
        # Avoid division by zero - all contributions negligible
        percentages = [0.0] * len(normalized_contributions)
    else:
        percentages = [c / total_normalized for c in normalized_contributions]

    return ContributionResult(
        vector=u,
        terms=terms,
        raw_contributions=raw_contributions,
        normalized_contributions=normalized_contributions,
        percentages=percentages,
        total_raw=total_raw,
        sigma=sigma,
        beta_term=beta_term,
        label=label,
    )
