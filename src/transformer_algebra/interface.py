"""Interface types for TransformerAlgebra ⟷ SymbolicTransformer communication.

This module defines reference types that describe HOW to retrieve data from
a HuggingFace model, rather than containing the data itself. Julia holds
these references and calls back to Python to resolve them when needed.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import hashlib
import json

import torch


# =============================================================================
# Core Reference Types
# =============================================================================

@dataclass
class ModelPath:
    """Path through a HuggingFace model object graph.

    Examples for GPT-NeoX (Pythia):
        ModelPath(["gpt_neox", "embed_in", "weight"])  # Embedding matrix
        ModelPath(["embed_out", "weight"])              # Unembedding matrix
        ModelPath(["gpt_neox", "layers", 5, "attention", "dense", "weight"])
    """
    segments: list[str | int]

    def resolve(self, model) -> torch.Tensor:
        """Navigate the model to get the tensor at this path."""
        obj = model
        for seg in self.segments:
            if isinstance(seg, int):
                obj = obj[seg]
            else:
                obj = getattr(obj, seg)
        return obj

    def to_dict(self) -> dict:
        return {"type": "ModelPath", "segments": self.segments}

    @classmethod
    def from_dict(cls, d: dict) -> "ModelPath":
        return cls(segments=d["segments"])


@dataclass
class TokenRef:
    """Reference to a token in the vocabulary.

    Attributes:
        token_id: Index in vocabulary
        text: String representation (e.g., " Dublin" with leading space)
    """
    token_id: int
    text: str

    def to_dict(self) -> dict:
        return {"type": "TokenRef", "token_id": self.token_id, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "TokenRef":
        return cls(token_id=d["token_id"], text=d["text"])


@dataclass
class EmbeddingRef:
    """Reference to an embedding vector (row of W_E).

    Notation: $\\underline{\\text{token}}$
    """
    token: TokenRef
    weights_path: ModelPath

    def resolve(self, model) -> torch.Tensor:
        W_E = self.weights_path.resolve(model)
        return W_E[self.token.token_id, :]

    def to_dict(self) -> dict:
        return {
            "type": "EmbeddingRef",
            "token": self.token.to_dict(),
            "weights_path": self.weights_path.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingRef":
        return cls(
            token=TokenRef.from_dict(d["token"]),
            weights_path=ModelPath.from_dict(d["weights_path"]),
        )


@dataclass
class UnembeddingRef:
    """Reference to an unembedding vector (row of W_U).

    Notation: $\\overline{\\text{token}}$
    """
    token: TokenRef
    weights_path: ModelPath

    def resolve(self, model) -> torch.Tensor:
        W_U = self.weights_path.resolve(model)
        return W_U[self.token.token_id, :]

    def to_dict(self) -> dict:
        return {
            "type": "UnembeddingRef",
            "token": self.token.to_dict(),
            "weights_path": self.weights_path.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UnembeddingRef":
        return cls(
            token=TokenRef.from_dict(d["token"]),
            weights_path=ModelPath.from_dict(d["weights_path"]),
        )


@dataclass
class LayerNormRef:
    """Reference to layer normalization parameters.

    LN(x) = γ * (x - μ) / σ + β
    """
    weight_path: ModelPath  # γ (scale)
    bias_path: ModelPath    # β (shift)
    epsilon: float = 1e-5

    def resolve_params(self, model) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Returns (γ, β, ε)."""
        γ = self.weight_path.resolve(model)
        β = self.bias_path.resolve(model)
        return γ, β, self.epsilon

    def to_dict(self) -> dict:
        return {
            "type": "LayerNormRef",
            "weight_path": self.weight_path.to_dict(),
            "bias_path": self.bias_path.to_dict(),
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerNormRef":
        return cls(
            weight_path=ModelPath.from_dict(d["weight_path"]),
            bias_path=ModelPath.from_dict(d["bias_path"]),
            epsilon=d.get("epsilon", 1e-5),
        )


@dataclass
class ResidualRef:
    """Reference to a residual vector in a cached context.

    Notation: $x^i_j$ = residual after layer i at position j
    """
    context_id: str
    layer: int      # 0 = after embedding, 1..n = after blocks
    position: int   # Token position (negative indexing supported)

    def to_dict(self) -> dict:
        return {
            "type": "ResidualRef",
            "context_id": self.context_id,
            "layer": self.layer,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResidualRef":
        return cls(
            context_id=d["context_id"],
            layer=d["layer"],
            position=d["position"],
        )


@dataclass
class BlockContribRef:
    """Reference to a transformer block's contribution.

    Notation: $\\Delta x^i_j$ = contribution from block i at position j
    """
    context_id: str
    block: int        # Block index (1..n_layers)
    position: int
    component: str    # "attention", "mlp", or "total"
    head: int | None = None  # For attention, specific head (None = sum of all)

    def to_dict(self) -> dict:
        return {
            "type": "BlockContribRef",
            "context_id": self.context_id,
            "block": self.block,
            "position": self.position,
            "component": self.component,
            "head": self.head,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BlockContribRef":
        return cls(
            context_id=d["context_id"],
            block=d["block"],
            position=d["position"],
            component=d["component"],
            head=d.get("head"),
        )


# =============================================================================
# Architecture Profiles
# =============================================================================

@dataclass
class ArchitectureProfile:
    """Maps abstract concepts to concrete model paths.

    Different HuggingFace model architectures have different internal structures.
    This profile allows the same interface to work across GPT-NeoX, Llama, etc.
    """
    name: str
    embedding_weights: ModelPath
    unembedding_weights: ModelPath
    final_ln_weight: ModelPath
    final_ln_bias: ModelPath
    final_ln_epsilon: float
    n_layers: int
    d_model: int
    n_heads: int

    # Functions to construct paths for specific layers
    block_ln_weight: Callable[[int], ModelPath] = field(repr=False)
    block_ln_bias: Callable[[int], ModelPath] = field(repr=False)

    def embedding_ref(self, token: TokenRef) -> EmbeddingRef:
        """Create an EmbeddingRef for a token."""
        return EmbeddingRef(token=token, weights_path=self.embedding_weights)

    def unembedding_ref(self, token: TokenRef) -> UnembeddingRef:
        """Create an UnembeddingRef for a token."""
        return UnembeddingRef(token=token, weights_path=self.unembedding_weights)

    def final_ln_ref(self) -> LayerNormRef:
        """Create a LayerNormRef for the final layer norm."""
        return LayerNormRef(
            weight_path=self.final_ln_weight,
            bias_path=self.final_ln_bias,
            epsilon=self.final_ln_epsilon,
        )

    def block_ln_ref(self, layer: int) -> LayerNormRef:
        """Create a LayerNormRef for a specific block's layer norm."""
        return LayerNormRef(
            weight_path=self.block_ln_weight(layer),
            bias_path=self.block_ln_bias(layer),
            epsilon=self.final_ln_epsilon,  # Usually same epsilon
        )


def gpt_neox_profile(config) -> ArchitectureProfile:
    """Create an ArchitectureProfile for GPT-NeoX models (Pythia)."""
    return ArchitectureProfile(
        name="gpt_neox",
        embedding_weights=ModelPath(["gpt_neox", "embed_in", "weight"]),
        unembedding_weights=ModelPath(["embed_out", "weight"]),
        final_ln_weight=ModelPath(["gpt_neox", "final_layer_norm", "weight"]),
        final_ln_bias=ModelPath(["gpt_neox", "final_layer_norm", "bias"]),
        final_ln_epsilon=config.layer_norm_eps,
        n_layers=config.num_hidden_layers,
        d_model=config.hidden_size,
        n_heads=config.num_attention_heads,
        block_ln_weight=lambda i: ModelPath(["gpt_neox", "layers", i, "input_layernorm", "weight"]),
        block_ln_bias=lambda i: ModelPath(["gpt_neox", "layers", i, "input_layernorm", "bias"]),
    )


# =============================================================================
# Context Cache
# =============================================================================

@dataclass
class ContextCache:
    """Cached intermediate states for a specific prompt.

    Created when the model processes a prompt. All intermediate residuals
    are cached and can be referenced by ResidualRef/BlockContribRef.
    """
    context_id: str
    prompt: str
    tokens: list[TokenRef]
    n_positions: int
    n_layers: int

    # Cached tensors (layer, position) -> tensor
    _residuals: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    _block_contribs: dict[tuple[int, int, str], torch.Tensor] = field(default_factory=dict)

    @staticmethod
    def generate_id(prompt: str) -> str:
        """Generate a unique context ID from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:12]

    def get_residual(self, layer: int, position: int) -> torch.Tensor:
        """Get cached residual, handling negative position indexing."""
        if position < 0:
            position = self.n_positions + position
        return self._residuals[(layer, position)]

    def set_residual(self, layer: int, position: int, tensor: torch.Tensor):
        """Cache a residual tensor."""
        self._residuals[(layer, position)] = tensor

    def get_block_contrib(self, block: int, position: int, component: str) -> torch.Tensor:
        """Get cached block contribution."""
        if position < 0:
            position = self.n_positions + position
        return self._block_contribs[(block, position, component)]

    def set_block_contrib(self, block: int, position: int, component: str, tensor: torch.Tensor):
        """Cache a block contribution tensor."""
        self._block_contribs[(block, position, component)] = tensor


# =============================================================================
# Serialization Helpers
# =============================================================================

def ref_to_dict(ref) -> dict:
    """Convert any reference type to a dict for JSON serialization."""
    return ref.to_dict()


def ref_from_dict(d: dict):
    """Reconstruct a reference from a dict."""
    type_map = {
        "ModelPath": ModelPath,
        "TokenRef": TokenRef,
        "EmbeddingRef": EmbeddingRef,
        "UnembeddingRef": UnembeddingRef,
        "LayerNormRef": LayerNormRef,
        "ResidualRef": ResidualRef,
        "BlockContribRef": BlockContribRef,
    }
    ref_type = d.get("type")
    if ref_type not in type_map:
        raise ValueError(f"Unknown reference type: {ref_type}")
    return type_map[ref_type].from_dict(d)


def refs_to_json(refs: list) -> str:
    """Serialize a list of references to JSON."""
    return json.dumps([ref_to_dict(r) for r in refs])


def refs_from_json(s: str) -> list:
    """Deserialize a list of references from JSON."""
    return [ref_from_dict(d) for d in json.loads(s)]
