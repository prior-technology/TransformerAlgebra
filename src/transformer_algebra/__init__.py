"""TransformerAlgebra - Symbolic notation and tools for analyzing transformer internal states."""

from .core import (
    # Protocol
    VectorLike,
    # Core classes
    PromptedTransformer,
    ModelConfig,
    load_pythia_model,
    EmbeddingVector,
    ResidualVector,
    BlockContribution,
    AttentionContribution,
    MLPContribution,
    VectorSum,
    LayerNormApplication,
    # Expression types for symbolic manipulation
    CenteredVector,
    GammaScaled,
    ScaledVector,
    InnerProduct,
    ScalarSum,
    ScalarValue,
    UnembeddingVector,
    # Logits
    LogitMapping,
    LogitValue,
    logits,
    # Predictions
    ProbabilityMapping,
    ProbabilityValue,
    predict,
    # Functions
    expand,
    # Contribution analysis
    ContributionResult,
    contribution,
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
    # Protocol
    "VectorLike",
    # Core classes
    "PromptedTransformer",
    "ModelConfig",
    "load_pythia_model",
    "EmbeddingVector",
    "ResidualVector",
    "BlockContribution",
    "AttentionContribution",
    "MLPContribution",
    "VectorSum",
    "LayerNormApplication",
    # Expression types for symbolic manipulation
    "CenteredVector",
    "GammaScaled",
    "ScaledVector",
    "InnerProduct",
    "ScalarSum",
    "ScalarValue",
    "UnembeddingVector",
    # Logits
    "LogitMapping",
    "LogitValue",
    "logits",
    # Predictions
    "ProbabilityMapping",
    "ProbabilityValue",
    "predict",
    # Functions
    "expand",
    # Contribution analysis
    "ContributionResult",
    "contribution",
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
