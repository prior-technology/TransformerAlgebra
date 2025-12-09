"""Core classes for analyzing transformer internal states."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
