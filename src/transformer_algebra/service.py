"""Service API for Julia to interact with the model.

This module provides the operations that SymbolicTransformer (Julia) calls
to resolve references and perform computations on the actual model.
"""

from dataclasses import dataclass
from typing import Any
import torch
import torch.nn.functional as F

from .interface import (
    ArchitectureProfile,
    TokenRef,
    EmbeddingRef,
    UnembeddingRef,
    LayerNormRef,
    ResidualRef,
    BlockContribRef,
    ContextCache,
    gpt_neox_profile,
    ref_from_dict,
)


class TransformerService:
    """Service that Julia calls to interact with the model.

    This class manages contexts (cached intermediate states) and provides
    methods to resolve references and perform computations.
    """

    def __init__(self, model, tokenizer):
        """Initialize the service with a HuggingFace model.

        Args:
            model: A HuggingFace causal LM
            tokenizer: The corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.profile = gpt_neox_profile(model.config)
        self.contexts: dict[str, ContextCache] = {}

    # =========================================================================
    # Token Operations
    # =========================================================================

    def tokenize(self, text: str) -> list[TokenRef]:
        """Tokenize text and return TokenRefs."""
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = tokens["input_ids"][0].tolist()
        return [
            TokenRef(token_id=tid, text=self.tokenizer.decode([tid]))
            for tid in token_ids
        ]

    def get_token_ref(self, text: str) -> TokenRef:
        """Get a TokenRef for text that should be a single token.

        Raises ValueError if text tokenizes to multiple tokens.
        """
        refs = self.tokenize(text)
        if len(refs) != 1:
            raise ValueError(
                f"'{text}' tokenizes to {len(refs)} tokens, expected 1. "
                f"Tokens: {[r.text for r in refs]}"
            )
        return refs[0]

    # =========================================================================
    # Context Management
    # =========================================================================

    def create_context(self, prompt: str, extract_contribs: bool = True) -> str:
        """Run model on prompt, cache all states, return context_id.

        Args:
            prompt: The prompt text to process
            extract_contribs: Whether to extract per-block contributions (slower)

        Returns:
            context_id: Unique identifier for this cached context
        """
        context_id = ContextCache.generate_id(prompt)

        # If already cached, return existing
        if context_id in self.contexts:
            return context_id

        # Tokenize
        tokens = self.tokenize(prompt)
        input_ids = torch.tensor([[t.token_id for t in tokens]])

        # Run model with hidden states output
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)

        # Create context cache
        n_positions = len(tokens)
        n_layers = self.profile.n_layers
        cache = ContextCache(
            context_id=context_id,
            prompt=prompt,
            tokens=tokens,
            n_positions=n_positions,
            n_layers=n_layers,
        )

        # Cache all residuals from hidden_states
        # hidden_states[0] = after embedding
        # hidden_states[i] = after block i (1-indexed in our notation)
        for layer_idx, hidden_state in enumerate(outputs.hidden_states):
            for pos in range(n_positions):
                cache.set_residual(layer_idx, pos, hidden_state[0, pos, :].clone())

        # Extract block contributions if requested
        if extract_contribs:
            self._extract_block_contributions(cache, input_ids)

        self.contexts[context_id] = cache
        return context_id

    def _extract_block_contributions(self, cache: ContextCache, input_ids: torch.Tensor):
        """Extract per-block contributions using hooks.

        This registers forward hooks to capture attention and MLP outputs.
        """
        # For GPT-NeoX, we need to hook into the parallel attention/MLP structure
        # This is a simplified version - full implementation would capture
        # attention outputs, MLP outputs, and per-head contributions

        # For now, compute block contributions as difference between residuals
        for block in range(1, cache.n_layers + 1):
            for pos in range(cache.n_positions):
                prev_residual = cache.get_residual(block - 1, pos)
                curr_residual = cache.get_residual(block, pos)
                contrib = curr_residual - prev_residual
                cache.set_block_contrib(block, pos, "total", contrib)

    def get_context_info(self, context_id: str) -> dict:
        """Get metadata about a cached context."""
        cache = self.contexts[context_id]
        return {
            "context_id": context_id,
            "prompt": cache.prompt,
            "tokens": [t.to_dict() for t in cache.tokens],
            "n_positions": cache.n_positions,
            "n_layers": cache.n_layers,
            "d_model": self.profile.d_model,
        }

    # =========================================================================
    # Reference Resolution
    # =========================================================================

    def resolve(self, ref: dict | Any) -> torch.Tensor:
        """Resolve any reference type to its tensor value.

        Args:
            ref: Either a reference object or a dict (from JSON)

        Returns:
            The resolved tensor
        """
        if isinstance(ref, dict):
            ref = ref_from_dict(ref)

        if isinstance(ref, EmbeddingRef):
            return ref.resolve(self.model)
        elif isinstance(ref, UnembeddingRef):
            return ref.resolve(self.model)
        elif isinstance(ref, ResidualRef):
            cache = self.contexts[ref.context_id]
            return cache.get_residual(ref.layer, ref.position)
        elif isinstance(ref, BlockContribRef):
            cache = self.contexts[ref.context_id]
            return cache.get_block_contrib(ref.block, ref.position, ref.component)
        elif isinstance(ref, LayerNormRef):
            γ, β, ε = ref.resolve_params(self.model)
            return {"weight": γ, "bias": β, "epsilon": ε}
        else:
            raise TypeError(f"Cannot resolve reference of type {type(ref)}")

    def resolve_to_list(self, ref: dict | Any) -> list[float]:
        """Resolve a reference and return as Python list (for JSON serialization)."""
        tensor = self.resolve(ref)
        if isinstance(tensor, dict):
            return {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in tensor.items()}
        return tensor.tolist()

    # =========================================================================
    # Computations
    # =========================================================================

    def inner_product(self, ref1: dict | Any, ref2: dict | Any) -> float:
        """Compute ⟨v1, v2⟩ for two vector refs."""
        v1 = self.resolve(ref1)
        v2 = self.resolve(ref2)
        return torch.dot(v1, v2).item()

    def apply_layer_norm(self, ln_ref: dict | Any, vector_ref: dict | Any) -> torch.Tensor:
        """Apply layer normalization to a vector."""
        if isinstance(ln_ref, dict):
            ln_ref = ref_from_dict(ln_ref)
        vector = self.resolve(vector_ref)

        γ, β, ε = ln_ref.resolve_params(self.model)

        # LN(x) = γ * (x - μ) / sqrt(var + ε) + β
        mean = vector.mean()
        var = vector.var(unbiased=False)
        normalized = (vector - mean) / torch.sqrt(var + ε)
        return γ * normalized + β

    def compute_logits(self, residual_ref: dict | Any) -> torch.Tensor:
        """Apply final LN + unembedding to get all logits."""
        residual = self.resolve(residual_ref)

        # Get final layer norm
        final_ln = self.profile.final_ln_ref()
        normed = self.apply_layer_norm(final_ln, residual_ref)

        # Apply unembedding (W_U @ normed)
        W_U = self.profile.unembedding_weights.resolve(self.model)
        return normed @ W_U.T

    def top_predictions(self, residual_ref: dict | Any, k: int = 10) -> list[dict]:
        """Get top k predictions with tokens and logits."""
        logits = self.compute_logits(residual_ref)
        probs = F.softmax(logits, dim=-1)

        top_logits, top_indices = torch.topk(logits, k)

        return [
            {
                "rank": i + 1,
                "token_id": idx.item(),
                "token_text": self.tokenizer.decode([idx.item()]),
                "logit": top_logits[i].item(),
                "probability": probs[idx].item(),
            }
            for i, idx in enumerate(top_indices)
        ]

    def logit_for_token(self, residual_ref: dict | Any, token_text: str) -> float:
        """Get logit for a specific token."""
        token_ref = self.get_token_ref(token_text)
        logits = self.compute_logits(residual_ref)
        return logits[token_ref.token_id].item()

    # =========================================================================
    # Gradient Operations (for future use)
    # =========================================================================

    def gradient(self, output_ref: dict | Any, input_ref: dict | Any) -> torch.Tensor:
        """Compute ∂output/∂input using torch autograd.

        This enables understanding how changes propagate through the model.
        Currently a placeholder for future implementation.
        """
        raise NotImplementedError("Gradient computation not yet implemented")

    # =========================================================================
    # Reference Factories
    # =========================================================================

    def embedding_ref(self, text: str) -> EmbeddingRef:
        """Create an EmbeddingRef for the given text (must be single token)."""
        token = self.get_token_ref(text)
        return self.profile.embedding_ref(token)

    def unembedding_ref(self, text: str) -> UnembeddingRef:
        """Create an UnembeddingRef for the given text (must be single token)."""
        token = self.get_token_ref(text)
        return self.profile.unembedding_ref(token)

    def residual_ref(self, context_id: str, layer: int, position: int = -1) -> ResidualRef:
        """Create a ResidualRef for a cached residual."""
        return ResidualRef(context_id=context_id, layer=layer, position=position)

    def block_contrib_ref(
        self,
        context_id: str,
        block: int,
        position: int = -1,
        component: str = "total",
    ) -> BlockContribRef:
        """Create a BlockContribRef for a block's contribution."""
        return BlockContribRef(
            context_id=context_id,
            block=block,
            position=position,
            component=component,
        )

    # =========================================================================
    # High-Level Analysis
    # =========================================================================

    def decompose_logit(
        self,
        context_id: str,
        token_text: str,
        position: int = -1,
    ) -> list[dict]:
        """Decompose a token's logit into per-block contributions.

        Returns a list of contributions from embedding and each block.
        This is the key analysis operation for interpretability.
        """
        cache = self.contexts[context_id]
        unembed_ref = self.unembedding_ref(token_text)
        unembed_vec = self.resolve(unembed_ref)

        # Get final layer norm params for scaling
        final_ln = self.profile.final_ln_ref()
        γ, β, ε = final_ln.resolve_params(self.model)

        # Get final residual for normalization scale
        final_residual = cache.get_residual(cache.n_layers, position)
        mean = final_residual.mean()
        var = final_residual.var(unbiased=False)
        scale = 1.0 / torch.sqrt(var + ε)

        results = []

        # Contribution from embedding (layer 0)
        embed_residual = cache.get_residual(0, position)
        centered_embed = embed_residual - embed_residual.mean()
        scaled_embed = γ * centered_embed * scale
        embed_contrib = torch.dot(unembed_vec, scaled_embed).item()
        results.append({
            "source": "embedding",
            "layer": 0,
            "contribution": embed_contrib,
            "label": f"⟨{token_text}̄, embed⟩",
        })

        # Contribution from each block
        for block in range(1, cache.n_layers + 1):
            block_contrib = cache.get_block_contrib(block, position, "total")
            centered_contrib = block_contrib - block_contrib.mean()
            scaled_contrib = γ * centered_contrib * scale
            contrib_value = torch.dot(unembed_vec, scaled_contrib).item()
            results.append({
                "source": f"block_{block}",
                "layer": block,
                "contribution": contrib_value,
                "label": f"⟨{token_text}̄, Δx^{block}⟩",
            })

        # Contribution from bias term
        bias_contrib = torch.dot(unembed_vec, β).item()
        results.append({
            "source": "ln_bias",
            "layer": None,
            "contribution": bias_contrib,
            "label": f"⟨{token_text}̄, β⟩",
        })

        return results
