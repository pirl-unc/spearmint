"""SPEARMINT (Stage 3 FiLM) model configuration."""

from transformers import PretrainedConfig

# Canonical assay vocabulary -- all other modules should import from here.
DEFAULT_ASSAY_TYPES = ["SPA", "Purified_Fluor", "Cellular_Fluor", "Other"]
DEFAULT_TEMP_C = 37.0

__all__ = ["SpearmintConfig", "DEFAULT_ASSAY_TYPES", "DEFAULT_TEMP_C"]


class SpearmintConfig(PretrainedConfig):
    """Configuration for SPEARMINT pMHC-I stability prediction model.

    This is the Stage 3 FiLM model: ESM2-650M backbone with cross-chain
    multimer attention, a trainable projection head, and Feature-wise
    Linear Modulation (FiLM) conditioned on assay type + temperature.
    """

    model_type = "spearmint"

    def __init__(
        self,
        # ESM2 backbone
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        token_dropout: bool = True,
        use_multimer: bool = True,
        # Projection head (from Stage 2)
        hidden_dim: int = 512,
        # FiLM conditioning
        num_assays: int = len(DEFAULT_ASSAY_TYPES),
        assay_emb_dim: int = 32,
        temp_emb_dim: int = 8,
        film_hidden_dim: int = 512,
        dropout: float = 0.0,
        # Assay/temperature metadata
        assay_types: list = None,
        default_temp_c: float = DEFAULT_TEMP_C,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.token_dropout = token_dropout
        self.use_multimer = use_multimer
        self.hidden_dim = hidden_dim
        self.num_assays = num_assays
        self.assay_emb_dim = assay_emb_dim
        self.temp_emb_dim = temp_emb_dim
        self.film_hidden_dim = film_hidden_dim
        self.dropout = dropout
        self.assay_types = assay_types or list(DEFAULT_ASSAY_TYPES)
        self.default_temp_c = default_temp_c
        super().__init__(**kwargs)
