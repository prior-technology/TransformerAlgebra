"""TransformerAlgebra - Symbolic notation and tools for analyzing transformer internal states."""

from .core import (
    PromptedTransformer,
    ModelConfig,
    load_pythia_model,
    EmbeddingVector,
    ResidualVector,
    LogitMapping,
    LogitValue,
    logits,
    ProbabilityMapping,
    ProbabilityValue,
    predict,
)
from .interface import (
    ModelPath,
    TokenRef,
    EmbeddingRef,
    UnembeddingRef,
    LayerNormRef,
    ResidualRef,
    BlockContribRef,
    ArchitectureProfile,
    ContextCache,
    gpt_neox_profile,
)
from .service import TransformerService

__all__ = [
    # Core
    "PromptedTransformer",
    "ModelConfig",
    "load_pythia_model",
    "EmbeddingVector",
    "ResidualVector",
    "LogitMapping",
    "LogitValue",
    "logits",
    "ProbabilityMapping",
    "ProbabilityValue",
    "predict",
    # Interface types
    "ModelPath",
    "TokenRef",
    "EmbeddingRef",
    "UnembeddingRef",
    "LayerNormRef",
    "ResidualRef",
    "BlockContribRef",
    "ArchitectureProfile",
    "ContextCache",
    "gpt_neox_profile",
    # Service
    "TransformerService",
]
