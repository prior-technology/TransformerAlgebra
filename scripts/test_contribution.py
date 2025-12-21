#!/usr/bin/env python
"""Test the contribution function with context-dependent predictions.

Tests the hypothesis from doc/contribution.md:
- Context-dependent predictions (e.g., "Dublin" after "The capital of Ireland is")
  should show high contribution from later blocks that integrate context
- Embedding-dominated predictions should show high contribution from the embedding
"""

import sys
sys.path.insert(0, 'src')

from transformer_algebra import (
    PromptedTransformer,
    load_pythia_model,
    expand,
    contribution,
    predict,
    logits,
)


def test_context_dependent(model, tokenizer, model_name: str):
    """Test context-dependent prediction: Ireland → Dublin."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Context-dependent case
    print("\n--- Context-Dependent Prediction ---")
    print("Prompt: 'The capital of Ireland is'")
    print("Target: ' Dublin'")

    T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
    x = T(" is")

    # Show what the model actually predicts
    probs = predict(x)
    print(f"\nTop 5 predictions:")
    print(probs.summary(5))

    # Get contribution breakdown for Dublin
    ex = expand(x)
    print(f"\nExpanded into {len(ex)} terms")

    dublin_logit = logits(x)[" Dublin"]
    contrib = contribution(ex, dublin_logit)
    print(f"\n{contrib.summary(10)}")

    # Show full breakdown
    print(f"\nFull contribution breakdown:")
    print(contrib)

    return contrib


def test_embedding_dominated(model, tokenizer, model_name: str):
    """Test embedding-dominated prediction: word continuation."""
    print(f"\n{'='*60}")
    print(f"Embedding-Dominated Test")
    print(f"{'='*60}")

    # Test with a word prefix that strongly predicts continuations
    print("\n--- Embedding-Dominated Prediction ---")
    print("Prompt: 'The un'")
    print("Target: 'der' (as in 'under')")

    T = PromptedTransformer(model, tokenizer, "The un")
    x = T("der")  # Test if "under" continuation is embedding-driven

    # Actually let's test the prediction AT "un"
    T2 = PromptedTransformer(model, tokenizer, "The")
    x2 = T2(" un")

    probs = predict(x2)
    print(f"\nTop 5 predictions after ' un':")
    print(probs.summary(5))

    # Pick the top predicted token for analysis
    top_token = probs.top_k(1)[0][0]
    print(f"\nAnalyzing contribution for top prediction: {top_token!r}")

    ex = expand(x2)
    top_logit = logits(x2)[top_token]
    contrib = contribution(ex, top_logit)
    print(f"\n{contrib.summary(10)}")

    return contrib


def compare_tokens(model, tokenizer):
    """Compare contributions for different target tokens from same residual."""
    print(f"\n{'='*60}")
    print(f"Comparing Different Target Tokens")
    print(f"{'='*60}")

    T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
    x = T(" is")
    ex = expand(x)

    tokens_to_compare = [" Dublin", " the", " a", " London", " not"]

    for token in tokens_to_compare:
        try:
            token_logit = logits(x)[token]
            contrib = contribution(ex, token_logit)
            # Get embedding percentage
            embed_pct = contrib.percentages[0] * 100
            # Get sum of later blocks (last 4)
            late_blocks = sum(contrib.percentages[-4:]) * 100
            print(f"\n{token!r:12} - embed: {embed_pct:5.1f}%, late blocks: {late_blocks:5.1f}%")
            print(f"  Top contributor: {repr(contrib.top_k(1)[0][0])}")
        except Exception as e:
            print(f"\n{token!r:12} - Error: {e}")


if __name__ == "__main__":
    print("Loading pythia-160m-deduped...")
    model, tokenizer = load_pythia_model("EleutherAI/pythia-160m-deduped")

    # Run tests
    test_context_dependent(model, tokenizer, "pythia-160m-deduped")
    test_embedding_dominated(model, tokenizer, "pythia-160m-deduped")
    compare_tokens(model, tokenizer)

    print("\n" + "="*60)
    print("Tests complete!")
