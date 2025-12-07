"""Logit lens utilities for analyzing transformer hidden states.

The logit lens technique applies the unembedding to intermediate residuals
to see what the model would predict at each layer.
"""

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
class TokenInfo:
    """Information about tokenized text."""
    token_ids: list[int]
    str_tokens: list[str]
    
    def __repr__(self):
        return f"TokenInfo(tokens={self.str_tokens}, ids={self.token_ids})"


@dataclass
class LogitLensResult:
    """Result of logit lens analysis for a single token."""
    target_token: str
    target_token_id: int
    logits_by_layer: list[float]
    top_predictions: list[tuple[str, float]]  # (token, logit) pairs
    
    def print_report(self):
        """Print a formatted report of the logit lens results."""
        print(f"Logit for '{self.target_token}' at each layer:")
        print(f"  After embedding: {self.logits_by_layer[0]:+.2f}")
        for layer in range(1, len(self.logits_by_layer)):
            print(f"  After block {layer}: {self.logits_by_layer[layer]:+.2f}")
        
        print(f"\nTop {len(self.top_predictions)} predictions after final layer:")
        for i, (token, logit) in enumerate(self.top_predictions):
            print(f"  {i+1}. '{token}' (logit: {logit:.2f})")


class LogitLens:
    """Apply the logit lens technique to analyze transformer predictions across layers.
    
    The logit lens applies the final layer norm and unembedding to intermediate
    hidden states, revealing how predictions evolve through the network.
    
    Example:
        >>> model, tokenizer = load_pythia_model()
        >>> lens = LogitLens(model, tokenizer)
        >>> result = lens.analyze("The capital of Ireland is", " Dublin")
        >>> result.print_report()
    """
    
    def __init__(self, model, tokenizer):
        """Initialize the logit lens with a model and tokenizer.
        
        Args:
            model: A HuggingFace causal LM with GPT-NeoX architecture
            tokenizer: The corresponding tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Access model components (GPT-NeoX architecture)
        self.final_ln = model.gpt_neox.final_layer_norm
        self.unembed = model.embed_out
    
    def tokenize(self, text: str) -> TokenInfo:
        """Tokenize text and return token information.
        
        Args:
            text: Text to tokenize
            
        Returns:
            TokenInfo with token IDs and string representations
        """
        tokens = self.tokenizer(text, return_tensors="pt")
        token_ids = tokens["input_ids"][0].tolist()
        str_tokens = [self.tokenizer.decode([t]) for t in token_ids]
        return TokenInfo(token_ids=token_ids, str_tokens=str_tokens)
    
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
    
    def get_hidden_states(self, prompt: str):
        """Run the model and return hidden states at all layers.
        
        Args:
            prompt: Input text
            
        Returns:
            tuple: (hidden_states, outputs) where hidden_states[i] is the 
                   residual after layer i (0 = after embedding)
        """
        tokens = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens, output_hidden_states=True)
        return outputs.hidden_states, outputs
    
    def compute_logits_at_layer(
        self, 
        hidden_state: torch.Tensor, 
        position: int = -1
    ) -> torch.Tensor:
        """Compute logits from a hidden state at a specific position.
        
        Args:
            hidden_state: Hidden state tensor [batch, seq_len, d_model]
            position: Position in sequence to analyze (default: last)
            
        Returns:
            Logits tensor [vocab_size]
        """
        resid = hidden_state[0, position, :]
        resid_normed = self.final_ln(resid)
        return self.unembed(resid_normed)
    
    def analyze(
        self, 
        prompt: str, 
        target_token: str,
        position: int = -1,
        top_k: int = 5
    ) -> LogitLensResult:
        """Perform logit lens analysis for a target token across all layers.
        
        Args:
            prompt: Input text
            target_token: Token to track (e.g., " Dublin" with leading space)
            position: Position in sequence to analyze (default: last)
            top_k: Number of top predictions to include
            
        Returns:
            LogitLensResult with logits at each layer and top predictions
        """
        target_id = self.get_token_id(target_token)
        hidden_states, outputs = self.get_hidden_states(prompt)
        
        # Compute target token logit at each layer
        logits_by_layer = []
        final_layer_logits = None
        with torch.no_grad():
            for hidden in hidden_states:
                logits = self.compute_logits_at_layer(hidden, position)
                logits_by_layer.append(logits[target_id].item())
                final_layer_logits = logits  # Keep the last one
        
        # Get top predictions from final layer (using logit lens, not model output)
        # Note: outputs.logits differs slightly due to precision/fusion differences
        top_tokens = torch.topk(final_layer_logits, top_k)
        top_predictions = [
            (self.tokenizer.decode([tok_id.item()]), logit.item())
            for logit, tok_id in zip(top_tokens.values, top_tokens.indices)
        ]
        
        return LogitLensResult(
            target_token=target_token,
            target_token_id=target_id,
            logits_by_layer=logits_by_layer,
            top_predictions=top_predictions
        )
    
    def plot_logits(
        self, 
        result: LogitLensResult, 
        prompt: str,
        figsize: tuple[int, int] = (10, 4)
    ):
        """Plot logit lens results.
        
        Args:
            result: LogitLensResult from analyze()
            prompt: The prompt used (for title)
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=figsize)
        x_labels = ["emb"] + [f"B{i}" for i in range(1, len(result.logits_by_layer))]
        plt.plot(range(len(result.logits_by_layer)), result.logits_by_layer, 'o-', markersize=8)
        plt.xlabel("Layer")
        plt.ylabel(f"'{result.target_token}' logit")
        plt.title(f"Logit Lens: '{result.target_token}' prediction across layers\nPrompt: \"{prompt}\"")
        plt.xticks(range(len(result.logits_by_layer)), x_labels)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
