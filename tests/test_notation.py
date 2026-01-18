"""Tests for notation consistency between documentation and implementation.

These tests verify that the implementation matches the notation defined in
doc/notation.md and produces output consistent with the expectations.

## Key Notation from notation.md

- T^n(x) = x + Σ ΔB^i(x^{i-1})  -- unnormalized residual (pre-LN)
- T(x) = LN^T(T^n(x))           -- full transformer output (post-LN)
- ΔB^i = block contribution operator
- Block indices: 1 to n (not 0 to n-1)

## Current Inconsistencies (flagged with xfail)

1. ResidualVector.__repr__ returns T(embed) but tensor is T^n(embed)
2. expand() doesn't show LN^T wrapper
3. Block indices start at 0 in code but 1 in notation
"""

import pytest


# =============================================================================
# Notation Consistency Tests (what SHOULD work per notation.md)
# =============================================================================

class TestNotationConsistency:
    """Tests that verify notation.md is correctly implemented."""

    @pytest.mark.slow
    def test_residual_repr_shows_full_transformer(self):
        """ResidualVector.__repr__ shows T(embed) for full transformer output.

        Per notation.md line 96-97:
        > T(x) = LN^T(T^n(x))

        The ResidualVector conceptually represents T(x), and expand() reveals
        the internal structure with LN^T wrapper.
        """
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")

        repr_str = repr(x)
        # Should show T(embed(...)) for full transformer
        assert "T(embed" in repr_str, f"Expected T(embed...), got: {repr_str}"

    @pytest.mark.slow
    def test_expand_shows_ln_wrapper(self):
        """Per notation.md line 114-115:
        T(x) = LN^T(T^n(x)) = LN^T(x + Σ ΔB^i(x^{i-1}))

        expand() should show LN^T wrapper around the sum.
        """
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand, LayerNormApplication,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        # Should return LayerNormApplication
        assert isinstance(expanded, LayerNormApplication), \
            f"expand() should return LayerNormApplication, got {type(expanded)}"

        repr_str = repr(expanded)
        # Should show LN^T wrapper around the sum
        assert "LN^T(" in repr_str, \
            f"Expanded form should show LN^T wrapper, got: {repr_str}"

    @pytest.mark.slow
    def test_block_indices_start_at_one(self):
        """Per notation.md line 107:
        T^n(x) = x + Σ_{i=1}^{n} ΔB^i(x^{i-1})

        Block indices should be 1 to n, not 0 to n-1.
        """
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        repr_str = repr(expanded)

        # First block should be ΔB^1, not ΔB^0
        assert "ΔB^1" in repr_str, \
            f"First block should be ΔB^1 per notation, got: {repr_str}"
        assert "ΔB^0" not in repr_str, \
            f"Should not have ΔB^0 (indices start at 1), got: {repr_str}"

    @pytest.mark.slow
    def test_expand_inner_is_vector_sum(self):
        """The inner expression of LN^T should be a VectorSum."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand, VectorSum,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        assert isinstance(expanded.inner, VectorSum), \
            f"inner should be VectorSum, got {type(expanded.inner)}"


# =============================================================================
# Current Behavior Tests (what currently works)
# =============================================================================

class TestBlockNotation:
    """Test ΔB^i notation with 1-based indices."""

    @pytest.mark.slow
    def test_block_contribution_repr_uses_delta_b(self):
        """BlockContribution.__repr__ should return ΔB^{layer} with 1-based index."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        # Access inner VectorSum, first block is at index 1 (index 0 is embedding)
        block = expanded.inner[1]
        # Uses 1-indexed per notation.md
        assert repr(block) == "ΔB^1", f"Expected 'ΔB^1', got '{repr(block)}'"

    @pytest.mark.slow
    def test_attention_contribution_repr(self):
        """AttentionContribution.__repr__ should return ΔB^{layer}_A."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        block_expanded = expanded.inner[1].expand()
        attn = block_expanded[0]
        assert repr(attn) == "ΔB^1_A", f"Expected 'ΔB^1_A', got '{repr(attn)}'"

    @pytest.mark.slow
    def test_mlp_contribution_repr(self):
        """MLPContribution.__repr__ should return ΔB^{layer}_M."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        block_expanded = expanded.inner[1].expand()
        mlp = block_expanded[1]
        assert repr(mlp) == "ΔB^1_M", f"Expected 'ΔB^1_M', got '{repr(mlp)}'"


class TestExpandedResidualFormat:
    """Test the format of expanded residual expressions."""

    @pytest.mark.slow
    def test_expand_residual_contains_ln_embedding_and_blocks(self):
        """expand(x) should show LN^T wrapper with embedding and block contributions."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The capital")
        x = T(" of")
        expanded = expand(x)

        repr_str = repr(expanded)

        # Should have LN^T wrapper
        assert "LN^T(" in repr_str, f"Missing LN^T wrapper in: {repr_str}"

        # Should contain embedding
        assert "embed(' of')" in repr_str, f"Missing embedding in: {repr_str}"

        # Should contain block contributions with ΔB notation (1-indexed)
        assert "ΔB^1" in repr_str, f"Missing ΔB^1 notation in: {repr_str}"

        # Should NOT contain old Δx notation
        assert "Δx^" not in repr_str, f"Found old Δx notation in: {repr_str}"

        # Should NOT have 0-indexed blocks
        assert "ΔB^0" not in repr_str, f"Found ΔB^0 (should be 1-indexed) in: {repr_str}"


class TestLogitNotation:
    """Test logit notation and formatting.

    Per notation.md line 132-133:
    z_t = ⟨t̄, LN^T(x^n)⟩
    """

    @pytest.mark.slow
    def test_logit_value_repr(self):
        """LogitValue.__repr__ should show <unembed(token), T(x)> = value."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, logits,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Hello")
        x = T(" world")
        L = logits(x)

        logit_value = L[" the"]
        repr_str = repr(logit_value)

        assert "unembed(' the')" in repr_str
        assert "T(embed(' world'))" in repr_str
        assert "=" in repr_str

    @pytest.mark.slow
    def test_logits_summary_format(self):
        """logits.summary() should show ranked predictions with logit values."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, logits,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Hello")
        x = T(" world")
        L = logits(x)

        summary = L.summary()

        assert "1." in summary
        assert "logit:" in summary or "logit=" in summary


class TestProbabilityNotation:
    """Test probability notation and formatting."""

    @pytest.mark.slow
    def test_probability_value_repr(self):
        """ProbabilityValue.__repr__ should show P(token | T(x)) = value%."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, predict,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Hello")
        x = T(" world")
        p = predict(x)

        prob_value = p[" the"]
        repr_str = repr(prob_value)

        assert "P(" in repr_str
        assert "%" in repr_str


class TestLayerNormExpansion:
    """Test layer norm expansion in notation.

    Per notation.md § Logit Decomposition:
    z_t = (1/σ) Σᵢ ⟨t̄_γ, P(xᵢ)⟩ + b_t

    where:
    - t̄_γ = γ^T ⊙ t̄  (scaled unembedding)
    - P(xᵢ) = xᵢ - x̄ᵢ·1  (mean-centered)
    - σ = ||P(x^n)||/√N  (std dev of total residual)
    - b_t = t̄ · β^T  (bias contribution)
    """

    @pytest.mark.slow
    def test_contribution_shows_layer_norm_params(self):
        """contribution() should expose σ (sigma) and β_t (beta term)."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand, logits, contribution,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The capital")
        x = T(" of")
        ex = expand(x)
        L = logits(x)
        logit_val = L[" the"]

        # contribution() takes the inner VectorSum
        contrib = contribution(ex.inner, logit_val)

        assert hasattr(contrib, 'sigma'), "ContributionResult should have sigma"
        assert hasattr(contrib, 'beta_term'), "ContributionResult should have beta_term"
        assert contrib.sigma > 0, f"sigma should be positive, got {contrib.sigma}"

    @pytest.mark.slow
    def test_contribution_summary_shows_layer_norm_info(self):
        """contribution.summary() should mention σ and β_t."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand, logits, contribution,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        ex = expand(x)
        L = logits(x)
        logit_val = L[" the"]

        # contribution() takes the inner VectorSum
        contrib = contribution(ex.inner, logit_val)
        summary = contrib.summary()

        assert "σ=" in summary or "sigma=" in summary, f"Missing σ in: {summary}"
        assert "β" in summary or "beta" in summary, f"Missing β in: {summary}"


class TestExpandOnLogitsAndProbabilities:
    """Test that expand() works on LogitValue and ProbabilityValue.

    Per notebook expectations, expand on logits should show:
    ⟨t̄, LN^T(embed + ΔB^1 + ΔB^2 + ... + ΔB^n)⟩

    And further expansion through LN should show:
    (1/σ)(⟨t̄_γ, P(embed)⟩ + ⟨t̄_γ, P(ΔB^1)⟩ + ...) + b_t
    """

    @pytest.mark.slow
    @pytest.mark.skip(reason="Not yet implemented - expand on LogitValue")
    def test_expand_logit_value(self):
        """expand(logit_value) should show LN expansion."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, logits, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The capital")
        x = T(" of")
        L = logits(x)
        logit_val = L[" the"]

        expanded = expand(logit_val)
        repr_str = repr(expanded)

        # Should show the expanded form with LN
        assert "LN" in repr_str or "ln" in repr_str
        assert "ΔB^" in repr_str

    @pytest.mark.slow
    @pytest.mark.skip(reason="Not yet implemented - expand on ProbabilityValue")
    def test_expand_probability_value(self):
        """expand(prob_value) should show softmax over LN expansion."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, predict, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "The capital")
        x = T(" of")
        p = predict(x)
        prob_val = p[" the"]

        expanded = expand(prob_val)
        repr_str = repr(expanded)

        assert "ΔB^" in repr_str


# =============================================================================
# Mathematical Accuracy Tests
# =============================================================================

class TestMathematicalAccuracy:
    """Test that expansions are mathematically accurate."""

    @pytest.mark.slow
    def test_inner_sum_equals_pre_ln_residual(self):
        """Inner VectorSum (embed + ΔB^1 + ... + ΔB^n) should equal pre-LN residual T^n(x)."""
        import torch
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test prompt")
        x = T(" token")

        expanded = expand(x)

        # The inner sum (without LN) should equal the pre-LN residual
        assert torch.allclose(expanded.inner.tensor, x.tensor, atol=1e-5), \
            "Inner sum doesn't equal pre-LN residual"

    @pytest.mark.slow
    def test_ln_application_equals_normed(self):
        """LN^T applied to inner sum should equal x.normed."""
        import torch
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test prompt")
        x = T(" token")

        expanded = expand(x)

        # expanded.tensor applies LN to inner sum
        assert torch.allclose(expanded.tensor, x.normed, atol=1e-5), \
            "LN^T(inner) doesn't equal x.normed"

    @pytest.mark.slow
    def test_attention_mlp_sum_to_block(self):
        """ΔB^i_A + ΔB^i_M should equal ΔB^i."""
        import torch
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")

        expanded = expand(x)
        block = expanded.inner[1]  # First block contribution (ΔB^1)

        block_expanded = block.expand()

        assert torch.allclose(block_expanded.tensor, block.tensor, atol=1e-5), \
            "ΔB^i_A + ΔB^i_M doesn't equal ΔB^i"

    @pytest.mark.slow
    def test_contribution_sums_to_logit(self):
        """Sum of contributions + beta should approximate the logit."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand, logits, contribution,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        ex = expand(x)
        L = logits(x)
        logit_val = L[" the"]

        # contribution() works with inner VectorSum
        contrib = contribution(ex.inner, logit_val)

        actual = logit_val.value
        computed = contrib.computed_logit

        assert abs(actual - computed) < 1.0, \
            f"Computed logit {computed} differs from actual {actual}"

    @pytest.mark.slow
    def test_number_of_inner_terms(self):
        """Inner VectorSum should have 1 (embed) + n_layers (blocks) terms."""
        from transformer_algebra import (
            load_pythia_model, PromptedTransformer, expand,
        )

        model, tokenizer = load_pythia_model("EleutherAI/pythia-14m")
        T = PromptedTransformer(model, tokenizer, "Test")
        x = T(" input")
        expanded = expand(x)

        expected = 1 + T.config.n_layers
        assert len(expanded.inner) == expected, \
            f"Expected {expected} inner terms, got {len(expanded.inner)}"
