"""Core classes for analyzing transformer internal states."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from typing import Self


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
    """

    def __init__(self, tensor: torch.Tensor, token_text: str, token_id: int,
                 transformer: "PromptedTransformer"):
        self.tensor = tensor
        self.token_text = token_text
        self.token_id = token_id
        self.transformer = transformer

    def __repr__(self):
        return f"embed({self.token_text!r})"


class ResidualVector:
    """A residual stream vector - the output of T acting on an embedding.

    Represents T^i(embed(token)) - the result after i transformer blocks.
    T is a nonlinear operator; T^n means the full transformer, T^5 means
    truncated after block 5.
    """

    def __init__(self, tensor: torch.Tensor, transformer: "PromptedTransformer",
                 layer: int, position: int, token_text: str):
        self.tensor = tensor
        self.transformer = transformer
        self.layer = layer
        self.position = position
        self.token_text = token_text

    def __repr__(self):
        n_layers = self.transformer.config.n_layers
        embed_str = f"embed({self.token_text!r})"
        if self.layer == n_layers:
            return f"T({embed_str})"
        else:
            return f"T^{self.layer}({embed_str})"

    @property
    def normed(self) -> torch.Tensor:
        """Apply final layer norm to get the normalized residual."""
        return self.transformer._final_ln(self.tensor)


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

        Args:
            token: Either a token string (e.g., " is") or an EmbeddingVector
            layer: Which layer's residual to return (-1 = final, default)

        Returns:
            ResidualVector representing T^layer(embed(token))

        Example:
            >>> T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
            >>> T(" is")           # shorthand
            >>> T(T.embed(" is"))  # explicit embedding
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
        residual_tensor = outputs.hidden_states[layer][0, -1, :]

        return ResidualVector(
            tensor=residual_tensor,
            transformer=self,
            layer=layer,
            position=-1,
            token_text=token_text,
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
