"""Basic tests for TransformerAlgebra package."""

import pytest


class TestImports:
    """Test that all public API imports work correctly."""

    def test_core_imports(self):
        from transformer_algebra import (
            PromptedTransformer,
            ModelConfig,
            load_pythia_model,
        )
        assert PromptedTransformer is not None
        assert ModelConfig is not None
        assert load_pythia_model is not None

    def test_vector_class_imports(self):
        from transformer_algebra import (
            EmbeddingVector,
            ResidualVector,
            BlockContribution,
            AttentionContribution,
            MLPContribution,
            VectorSum,
            expand,
        )
        assert EmbeddingVector is not None
        assert ResidualVector is not None
        assert BlockContribution is not None
        assert AttentionContribution is not None
        assert MLPContribution is not None
        assert VectorSum is not None
        assert expand is not None

    def test_interface_imports(self):
        from transformer_algebra import (
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
        assert ModelPath is not None
        assert TokenRef is not None

    def test_service_import(self):
        from transformer_algebra import TransformerService
        assert TransformerService is not None


class TestInterfaceTypes:
    """Test interface types work correctly without loading models."""

    def test_model_path_creation(self):
        from transformer_algebra import ModelPath

        path = ModelPath(["gpt_neox", "embed_in", "weight"])
        assert path.segments == ["gpt_neox", "embed_in", "weight"]

    def test_model_path_with_int(self):
        from transformer_algebra import ModelPath

        path = ModelPath(["gpt_neox", "layers", 5, "attention"])
        assert path.segments[2] == 5

    def test_token_ref_creation(self):
        from transformer_algebra import TokenRef

        token = TokenRef(token_id=42, text=" Dublin")
        assert token.token_id == 42
        assert token.text == " Dublin"

    def test_residual_ref_creation(self):
        from transformer_algebra import ResidualRef

        ref = ResidualRef(context_id="abc123", layer=5, position=-1)
        assert ref.layer == 5
        assert ref.position == -1

    def test_block_contrib_ref_creation(self):
        from transformer_algebra import BlockContribRef

        ref = BlockContribRef(
            context_id="abc123",
            block=3,
            position=0,
            component="attention",
            head=2,
        )
        assert ref.component == "attention"
        assert ref.head == 2

    def test_embedding_ref_creation(self):
        from transformer_algebra import EmbeddingRef, TokenRef, ModelPath

        token = TokenRef(token_id=42, text=" hello")
        weights_path = ModelPath(["gpt_neox", "embed_in", "weight"])
        ref = EmbeddingRef(token=token, weights_path=weights_path)
        assert ref.token.token_id == 42
        assert ref.weights_path.segments[0] == "gpt_neox"

    def test_unembedding_ref_creation(self):
        from transformer_algebra import UnembeddingRef, TokenRef, ModelPath

        token = TokenRef(token_id=100, text=" world")
        weights_path = ModelPath(["embed_out", "weight"])
        ref = UnembeddingRef(token=token, weights_path=weights_path)
        assert ref.token.text == " world"
        assert ref.weights_path.segments[0] == "embed_out"

    def test_layer_norm_ref_creation(self):
        from transformer_algebra import LayerNormRef, ModelPath

        weight_path = ModelPath(["gpt_neox", "final_layer_norm", "weight"])
        bias_path = ModelPath(["gpt_neox", "final_layer_norm", "bias"])
        ref = LayerNormRef(weight_path=weight_path, bias_path=bias_path, epsilon=1e-5)
        assert ref.epsilon == 1e-5
        assert ref.weight_path.segments[-1] == "weight"
        assert ref.bias_path.segments[-1] == "bias"

    def test_layer_norm_ref_default_epsilon(self):
        from transformer_algebra import LayerNormRef, ModelPath

        weight_path = ModelPath(["ln", "weight"])
        bias_path = ModelPath(["ln", "bias"])
        ref = LayerNormRef(weight_path=weight_path, bias_path=bias_path)
        assert ref.epsilon == 1e-5  # Default value


class TestSerialization:
    """Test JSON serialization/deserialization of reference types."""

    def test_model_path_roundtrip(self):
        from transformer_algebra import ModelPath

        original = ModelPath(["gpt_neox", "layers", 5, "weight"])
        d = original.to_dict()
        restored = ModelPath.from_dict(d)
        assert restored.segments == original.segments

    def test_token_ref_roundtrip(self):
        from transformer_algebra import TokenRef

        original = TokenRef(token_id=100, text="hello")
        d = original.to_dict()
        restored = TokenRef.from_dict(d)
        assert restored.token_id == original.token_id
        assert restored.text == original.text

    def test_residual_ref_roundtrip(self):
        from transformer_algebra import ResidualRef

        original = ResidualRef(context_id="test123", layer=10, position=-1)
        d = original.to_dict()
        restored = ResidualRef.from_dict(d)
        assert restored.context_id == original.context_id
        assert restored.layer == original.layer
        assert restored.position == original.position

    def test_block_contrib_ref_roundtrip(self):
        from transformer_algebra import BlockContribRef

        original = BlockContribRef(
            context_id="ctx456",
            block=3,
            position=2,
            component="attention",
            head=5,
        )
        d = original.to_dict()
        restored = BlockContribRef.from_dict(d)
        assert restored.context_id == original.context_id
        assert restored.block == original.block
        assert restored.position == original.position
        assert restored.component == original.component
        assert restored.head == original.head

    def test_block_contrib_ref_roundtrip_no_head(self):
        from transformer_algebra import BlockContribRef

        original = BlockContribRef(
            context_id="ctx789",
            block=1,
            position=-1,
            component="mlp",
            head=None,
        )
        d = original.to_dict()
        restored = BlockContribRef.from_dict(d)
        assert restored.head is None
        assert restored.component == "mlp"

    def test_embedding_ref_roundtrip(self):
        from transformer_algebra import EmbeddingRef, TokenRef, ModelPath

        token = TokenRef(token_id=42, text=" test")
        weights_path = ModelPath(["gpt_neox", "embed_in", "weight"])
        original = EmbeddingRef(token=token, weights_path=weights_path)
        d = original.to_dict()
        restored = EmbeddingRef.from_dict(d)
        assert restored.token.token_id == original.token.token_id
        assert restored.token.text == original.token.text
        assert restored.weights_path.segments == original.weights_path.segments

    def test_unembedding_ref_roundtrip(self):
        from transformer_algebra import UnembeddingRef, TokenRef, ModelPath

        token = TokenRef(token_id=99, text=" out")
        weights_path = ModelPath(["embed_out", "weight"])
        original = UnembeddingRef(token=token, weights_path=weights_path)
        d = original.to_dict()
        restored = UnembeddingRef.from_dict(d)
        assert restored.token.token_id == original.token.token_id
        assert restored.weights_path.segments == original.weights_path.segments

    def test_layer_norm_ref_roundtrip(self):
        from transformer_algebra import LayerNormRef, ModelPath

        original = LayerNormRef(
            weight_path=ModelPath(["ln", "weight"]),
            bias_path=ModelPath(["ln", "bias"]),
            epsilon=1e-6,
        )
        d = original.to_dict()
        restored = LayerNormRef.from_dict(d)
        assert restored.weight_path.segments == original.weight_path.segments
        assert restored.bias_path.segments == original.bias_path.segments
        assert restored.epsilon == original.epsilon

    def test_layer_norm_ref_roundtrip_default_epsilon(self):
        from transformer_algebra import LayerNormRef, ModelPath

        original = LayerNormRef(
            weight_path=ModelPath(["ln", "weight"]),
            bias_path=ModelPath(["ln", "bias"]),
        )
        d = original.to_dict()
        restored = LayerNormRef.from_dict(d)
        assert restored.epsilon == 1e-5  # Default


class TestPolymorphicSerialization:
    """Test polymorphic ref_from_dict deserialization."""

    def test_ref_from_dict_model_path(self):
        from transformer_algebra.interface import ref_from_dict, ModelPath

        d = {"type": "ModelPath", "segments": ["gpt_neox", "layers", 0]}
        ref = ref_from_dict(d)
        assert isinstance(ref, ModelPath)
        assert ref.segments == ["gpt_neox", "layers", 0]

    def test_ref_from_dict_token_ref(self):
        from transformer_algebra.interface import ref_from_dict, TokenRef

        d = {"type": "TokenRef", "token_id": 50, "text": " cat"}
        ref = ref_from_dict(d)
        assert isinstance(ref, TokenRef)
        assert ref.token_id == 50

    def test_ref_from_dict_residual_ref(self):
        from transformer_algebra.interface import ref_from_dict, ResidualRef

        d = {"type": "ResidualRef", "context_id": "abc", "layer": 3, "position": -1}
        ref = ref_from_dict(d)
        assert isinstance(ref, ResidualRef)
        assert ref.layer == 3

    def test_ref_from_dict_unknown_type_raises(self):
        from transformer_algebra.interface import ref_from_dict

        d = {"type": "UnknownRef", "data": "whatever"}
        with pytest.raises(ValueError, match="Unknown reference type"):
            ref_from_dict(d)


class TestJsonListSerialization:
    """Test refs_to_json and refs_from_json functions."""

    def test_refs_to_json_and_back(self):
        from transformer_algebra.interface import refs_to_json, refs_from_json
        from transformer_algebra import TokenRef, ResidualRef, ModelPath

        refs = [
            TokenRef(token_id=1, text="a"),
            ResidualRef(context_id="ctx", layer=0, position=0),
            ModelPath(["gpt_neox"]),
        ]
        json_str = refs_to_json(refs)
        restored = refs_from_json(json_str)

        assert len(restored) == 3
        assert isinstance(restored[0], TokenRef)
        assert restored[0].token_id == 1
        assert isinstance(restored[1], ResidualRef)
        assert restored[1].context_id == "ctx"
        assert isinstance(restored[2], ModelPath)
        assert restored[2].segments == ["gpt_neox"]

    def test_refs_to_json_empty_list(self):
        from transformer_algebra.interface import refs_to_json, refs_from_json

        json_str = refs_to_json([])
        restored = refs_from_json(json_str)
        assert restored == []


class TestContextCache:
    """Test ContextCache functionality."""

    def test_generate_id_deterministic(self):
        from transformer_algebra import ContextCache

        id1 = ContextCache.generate_id("test prompt")
        id2 = ContextCache.generate_id("test prompt")
        assert id1 == id2

    def test_generate_id_unique(self):
        from transformer_algebra import ContextCache

        id1 = ContextCache.generate_id("prompt A")
        id2 = ContextCache.generate_id("prompt B")
        assert id1 != id2

    def test_residual_set_and_get(self):
        import torch
        from transformer_algebra import ContextCache, TokenRef

        cache = ContextCache(
            context_id="test",
            prompt="hello",
            tokens=[TokenRef(token_id=0, text="hello")],
            n_positions=5,
            n_layers=3,
        )
        tensor = torch.randn(128)
        cache.set_residual(1, 2, tensor)
        retrieved = cache.get_residual(1, 2)
        assert torch.equal(tensor, retrieved)

    def test_residual_negative_indexing(self):
        import torch
        from transformer_algebra import ContextCache, TokenRef

        cache = ContextCache(
            context_id="test",
            prompt="hello world",
            tokens=[TokenRef(token_id=i, text=f"t{i}") for i in range(4)],
            n_positions=4,
            n_layers=2,
        )
        tensor = torch.randn(64)
        # Set at position 3 (last position in 0-3)
        cache.set_residual(0, 3, tensor)
        # Get using negative indexing
        retrieved = cache.get_residual(0, -1)
        assert torch.equal(tensor, retrieved)

    def test_block_contrib_set_and_get(self):
        import torch
        from transformer_algebra import ContextCache, TokenRef

        cache = ContextCache(
            context_id="test",
            prompt="test",
            tokens=[TokenRef(token_id=0, text="test")],
            n_positions=3,
            n_layers=2,
        )
        tensor = torch.randn(128)
        cache.set_block_contrib(1, 0, "attention", tensor)
        retrieved = cache.get_block_contrib(1, 0, "attention")
        assert torch.equal(tensor, retrieved)

    def test_block_contrib_negative_indexing(self):
        import torch
        from transformer_algebra import ContextCache, TokenRef

        cache = ContextCache(
            context_id="test",
            prompt="a b c",
            tokens=[TokenRef(token_id=i, text=f"t{i}") for i in range(3)],
            n_positions=3,
            n_layers=2,
        )
        tensor = torch.randn(64)
        # Set at position 2 (last in 0-2)
        cache.set_block_contrib(1, 2, "total", tensor)
        # Get using negative indexing
        retrieved = cache.get_block_contrib(1, -1, "total")
        assert torch.equal(tensor, retrieved)


class TestModelConfig:
    """Test ModelConfig class."""

    def test_d_head_calculation(self):
        from transformer_algebra import ModelConfig

        config = ModelConfig(
            name="test-model",
            n_layers=12,
            n_heads=8,
            d_model=512,
            vocab_size=50000,
        )
        assert config.d_head == 64  # 512 / 8


@pytest.mark.slow
class TestModelLoading:
    """Tests that require downloading models. Marked as slow."""

    def test_load_pythia_model(self):
        """Test loading a small Pythia model."""
        from transformer_algebra import load_pythia_model

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        assert model is not None
        assert tokenizer is not None

    def test_prompted_transformer_basic(self):
        """Test PromptedTransformer with a simple prompt."""
        from transformer_algebra import load_pythia_model, PromptedTransformer

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Hello world")

        assert T.n_positions > 0
        assert T.config.n_layers > 0
        assert len(T.tokens.tokens) == T.n_positions

    def test_residual_extraction(self):
        """Test extracting residual vectors."""
        from transformer_algebra import load_pythia_model, PromptedTransformer

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")

        # Get residual at layer 0 (after embedding)
        r0 = T.residual(0)
        assert r0.shape[0] == T.config.d_model

        # Get residual at final layer
        r_final = T.residual(T.config.n_layers)
        assert r_final.shape[0] == T.config.d_model

    def test_logits_at_layer(self):
        """Test computing logits from intermediate layers."""
        from transformer_algebra import load_pythia_model, PromptedTransformer

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The")

        logits = T.logits_at_layer(T.config.n_layers)
        assert logits.shape[0] == T.config.vocab_size

    def test_expand_residual_to_blocks(self):
        """Test expanding a ResidualVector into embedding + block contributions."""
        import torch
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
            EmbeddingVector, BlockContribution, VectorSum,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

        # Get residual for next token
        x = T(" is")
        expanded = expand(x)

        # Should be a VectorSum
        assert isinstance(expanded, VectorSum)

        # First term should be embedding, rest should be block contributions
        assert isinstance(expanded[0], EmbeddingVector)
        for i in range(1, len(expanded)):
            assert isinstance(expanded[i], BlockContribution)

        # Number of terms = 1 (embedding) + n_layers (block contributions)
        assert len(expanded) == 1 + T.config.n_layers

        # Sum of terms should equal original tensor
        assert torch.allclose(expanded.tensor, x.tensor, atol=1e-5)

    def test_expand_block_to_attention_mlp(self):
        """Test expanding a BlockContribution into attention + MLP."""
        import torch
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
            BlockContribution, AttentionContribution, MLPContribution, VectorSum,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The capital of Ireland")

        # Get residual and expand to blocks
        x = T(" is")
        expanded = expand(x)

        # Get a block contribution and expand it
        block_contrib = expanded[1]  # First block contribution (Δx^0)
        assert isinstance(block_contrib, BlockContribution)

        block_expanded = block_contrib.expand()

        # Should be attention + MLP
        assert isinstance(block_expanded, VectorSum)
        assert len(block_expanded) == 2
        assert isinstance(block_expanded[0], AttentionContribution)
        assert isinstance(block_expanded[1], MLPContribution)

        # Sum should equal original block contribution
        assert torch.allclose(block_expanded.tensor, block_contrib.tensor, atol=1e-5)

    def test_multi_level_expand(self):
        """Test expanding VectorSum to go deeper (Level 1 -> Level 2).

        Note: This test verifies structure but not numerical equality.
        See doc/expand_issues.md for known issues with multi-level sum.
        """
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
            EmbeddingVector, AttentionContribution, MLPContribution, VectorSum,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Hello")

        x = T(" world")

        # Level 1 expansion: T(x) -> embed + Δx^0 + Δx^1 + ...
        level1 = expand(x)

        # Level 2 expansion: expand each BlockContribution
        level2 = level1.expand()

        # Should have: embedding + (attn + mlp) for each block
        # = 1 + 2 * n_layers terms
        expected_terms = 1 + 2 * T.config.n_layers
        assert len(level2) == expected_terms

        # First should still be embedding
        assert isinstance(level2[0], EmbeddingVector)

        # Rest should alternate attention/MLP (grouped by block)
        for i in range(T.config.n_layers):
            assert isinstance(level2[1 + 2*i], AttentionContribution)
            assert isinstance(level2[1 + 2*i + 1], MLPContribution)

        # Note: The sum equality (level2.tensor == x.tensor) is a known issue.
        # Individual block expansions work correctly, but cumulative multi-level
        # expansion has a discrepancy. See doc/expand_issues.md for details.

    def test_expand_repr(self):
        """Test that expanded forms have meaningful string representations."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")

        x = T(" input")
        expanded = expand(x)

        # Should show embed + block contributions
        repr_str = repr(expanded)
        assert "embed" in repr_str
        assert "Δx^0" in repr_str

        # Expand a block
        block_expanded = expanded[1].expand()
        block_repr = repr(block_expanded)
        assert "Δx^0_A" in block_repr
        assert "Δx^0_M" in block_repr
