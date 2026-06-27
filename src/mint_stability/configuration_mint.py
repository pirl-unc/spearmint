"""MINT Stability model configuration (S1/S2)."""

from transformers import PretrainedConfig

__all__ = ["MintStabilityConfig"]


class MintStabilityConfig(PretrainedConfig):
    """Configuration for MINT pMHC-I stability prediction model.

    This model uses a custom ESM2-650M backbone (with optional cross-chain
    multimer attention from MINT pretraining) and a projection head for
    regression on complex half-life.
    """

    model_type = "mint_stability"

    def __init__(
        self,
        # ESM2 backbone
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        token_dropout: bool = True,
        use_multimer: bool = True,
        # Projection head
        hidden_dim: int = 512,
        dropout: float = 0.2,
        output_size: int = 1,
        sigmoid_output: bool = False,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.token_dropout = token_dropout
        self.use_multimer = use_multimer
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.output_size = output_size
        self.sigmoid_output = sigmoid_output
        super().__init__(**kwargs)
