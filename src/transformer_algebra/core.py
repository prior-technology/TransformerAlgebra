"""Core classes for analyzing transformer internal states."""

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING, Protocol, runtime_checkable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from typing import Self


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

    @property
    def normed(self) -> torch.Tensor:
        """Apply final layer norm to get the normalized residual."""
        return self.transformer._final_ln(self._tensor)

    def expand(self) -> "VectorSum":
        """Expand into embedding plus block contributions.

        T(x) -> x + Δx^0 + Δx^1 + ... + Δx^{n-1}

        Returns:
            VectorSum of embedding and block contributions
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

        # Block contributions: Δx^i = hidden_states[i+1] - hidden_states[i]
        for i in range(self.layer):
            delta = (self._hidden_states[i + 1][0, self.position, :] -
                     self._hidden_states[i][0, self.position, :])
            terms.append(BlockContribution(
                tensor=delta,
                layer=i,
                transformer=self.transformer,
                token_text=self.token_text,
                position=self.position,
                hidden_states=self._hidden_states,
            ))

        return VectorSum(terms)


class AttentionContribution:
    """Contribution from the attention sublayer of a block.

    Represents Δx^i_A - the attention component of block i's contribution.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, layer: int,
                 transformer: "PromptedTransformer", token_text: str,
                 position: int):
        self._tensor = tensor
        self.layer = layer
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
        return f"Δx^{self.layer}_A"


class MLPContribution:
    """Contribution from the MLP sublayer of a block.

    Represents Δx^i_M - the MLP component of block i's contribution.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, layer: int,
                 transformer: "PromptedTransformer", token_text: str,
                 position: int):
        self._tensor = tensor
        self.layer = layer
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
        return f"Δx^{self.layer}_M"


class BlockContribution:
    """Contribution from a single transformer block.

    Represents Δx^i = T^{i+1}(x) - T^i(x), the change in residual
    from block i. Can be expanded into Δx^i_A + Δx^i_M.

    Implements VectorLike protocol.
    """

    def __init__(self, tensor: torch.Tensor, layer: int,
                 transformer: "PromptedTransformer", token_text: str,
                 position: int,
                 hidden_states: tuple[torch.Tensor, ...] | None = None,
                 attention_tensor: torch.Tensor | None = None,
                 mlp_tensor: torch.Tensor | None = None):
        self._tensor = tensor
        self.layer = layer
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
        return f"Δx^{self.layer}"

    def expand(self) -> "VectorSum":
        """Expand into attention + MLP contributions.

        Δx^i -> Δx^i_A + Δx^i_M

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

    Represents an expanded form like: x + Δx^0 + Δx^1 + ... + Δx^{n-1}

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
        return " + ".join(repr(t) for t in self.terms)

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


def expand(x: VectorLike) -> VectorSum:
    """Expand a vector into its additive components.

    For ResidualVector: T(x) -> embed(x) + Δx^0 + Δx^1 + ... + Δx^{n-1}
    For VectorSum: expand each term recursively
    For other VectorLike: wrap in single-term VectorSum

    Args:
        x: Any vector-like object

    Returns:
        VectorSum of components

    Example:
        >>> x = T(" is")
        >>> ex = expand(x)
        >>> print(ex)  # embed(' is') + Δx^0 + Δx^1 + ... + Δx^{11}
        >>> print(len(ex))  # 13 (embedding + 12 blocks)
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

class ContributionResult:
    """Result of contribution analysis showing how each term affects an inner product.

    Displays how terms in an expanded vector sum contribute to an inner product
    like a logit ⟨unembed(token), LN(x)⟩.
    """

    def __init__(
        self,
        inner_product: "LogitValue",
        terms: list[VectorLike],
        raw_contributions: list[float],
        percentages: list[float],
        total_raw: float,
    ):
        self.inner_product = inner_product
        self.terms = terms
        self.raw_contributions = raw_contributions
        self.percentages = percentages
        self.total_raw = total_raw

    @property
    def target_token(self) -> str:
        """The target token (for logit inner products)."""
        return self.inner_product.token_text

    def __repr__(self):
        lines = [f"Contributions to {self.inner_product!r}:"]
        for term, raw, pct in zip(self.terms, self.raw_contributions, self.percentages):
            sign = "+" if raw >= 0 else ""
            lines.append(f"  {repr(term)}: {sign}{raw:.2f} ({pct:+.1%})")
        lines.append(f"  Total: {self.total_raw:.2f}")
        return "\n".join(lines)

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        """Iterate over (term, raw_contribution, percentage) tuples."""
        return zip(self.terms, self.raw_contributions, self.percentages)

    def top_k(self, k: int = 5) -> list[tuple[VectorLike, float, float]]:
        """Return the k terms with largest absolute contribution."""
        items = list(zip(self.terms, self.raw_contributions, self.percentages))
        items.sort(key=lambda x: abs(x[1]), reverse=True)
        return items[:k]

    def summary(self, k: int = 5) -> str:
        """Return a formatted summary of top contributions."""
        lines = [f"Top {k} contributions to {self.target_token!r}:"]
        for term, raw, pct in self.top_k(k):
            sign = "+" if raw >= 0 else ""
            lines.append(f"  {repr(term)}: {sign}{raw:.2f} ({pct:+.1%})")
        return "\n".join(lines)


def contribution(expanded: VectorSum, inner_product: LogitValue) -> ContributionResult:
    """Compute how each term in an expanded vector contributes to an inner product.

    Given an expanded residual (from expand()) and an inner product (like a logit),
    computes each term's contribution to the final value.

    The inner product ⟨u, LN(Σᵢ xᵢ)⟩ decomposes as:
        value = (1/σ) * Σᵢ cᵢ + bias
    where cᵢ = ⟨u ⊙ γ, xᵢ - μᵢ⟩

    See doc/contribution.md and doc/analysis/speculation.md for mathematical details.

    Args:
        expanded: A VectorSum from expand(residual)
        inner_product: A LogitValue from logits(x)[token]

    Returns:
        ContributionResult with breakdown by term

    Example:
        >>> T = PromptedTransformer(model, tokenizer, "The capital of Ireland")
        >>> x = T(" is")
        >>> ex = expand(x)
        >>> logit = logits(x)[" Dublin"]
        >>> contrib = contribution(ex, logit)
        >>> print(contrib.summary())
    """
    if not isinstance(expanded, VectorSum):
        raise TypeError(f"Expected VectorSum, got {type(expanded).__name__}")

    if not isinstance(inner_product, LogitValue):
        raise TypeError(
            f"Expected LogitValue (from logits(x)[token]), got {type(inner_product).__name__}"
        )

    if len(expanded.terms) == 0:
        raise ValueError("VectorSum has no terms")

    # Get transformer from the inner product's residual
    transformer = inner_product._residual.transformer

    # Get layer norm parameters (γ, β)
    final_ln = transformer._final_ln
    gamma = final_ln.weight.detach()  # [d_model]
    # beta = final_ln.bias.detach()  # [d_model] - constant term, reported separately

    # Get unembedding vector for target token
    token_id = transformer.get_token_id(inner_product.token_text)
    unembed_matrix = transformer._unembed.weight  # [vocab_size, d_model]
    u = unembed_matrix[token_id, :].detach()  # [d_model]

    # Precompute u ⊙ γ (element-wise product)
    u_gamma = u * gamma  # [d_model]

    # Compute contribution for each term
    raw_contributions = []
    for term in expanded.terms:
        x_i = term.tensor.detach()  # [d_model]
        mu_i = x_i.mean()  # scalar
        centered = x_i - mu_i  # [d_model]
        c_i = torch.dot(u_gamma, centered).item()  # scalar
        raw_contributions.append(c_i)

    # Compute total and percentages
    total_raw = sum(raw_contributions)
    if abs(total_raw) < 1e-10:
        # Avoid division by zero - all contributions negligible
        percentages = [0.0] * len(raw_contributions)
    else:
        percentages = [c / total_raw for c in raw_contributions]

    return ContributionResult(
        inner_product=inner_product,
        terms=list(expanded.terms),
        raw_contributions=raw_contributions,
        percentages=percentages,
        total_raw=total_raw,
    )
