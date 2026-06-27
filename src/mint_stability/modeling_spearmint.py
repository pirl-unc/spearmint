"""SPEARMINT: Stage 3 FiLM stability model."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ESM2
from .configuration_spearmint import SpearmintConfig
from .modeling_base import StabilityPreTrainedModel

__all__ = ["SpearmintPreTrainedModel", "SpearmintForStabilityPrediction"]


class SpearmintPreTrainedModel(StabilityPreTrainedModel):
    """Base class for SPEARMINT models."""

    config_class = SpearmintConfig


class SpearmintForStabilityPrediction(SpearmintPreTrainedModel):
    """SPEARMINT: Stage 3 FiLM stability model.

    Architecture:
        r       = MeanPool(ESM2(x))                     frozen backbone
        h       = ReLU(W1 r + b1)                       trainable projection
        u       = [Embed(assay); Linear(temp); h]        conditioning input
        [g, b]  = MLP_film(u)                            FiLM parameters
        h_mod   = g * h + b                              modulation
        y_hat   = readout(h_mod)                         scalar output

    Output is in log1p(hours) scale. Apply expm1() to get hours.
    """

    config_class = SpearmintConfig
    _keys_to_ignore_on_load_missing = ["esm.lm_head.weight"]

    def __init__(self, config: SpearmintConfig):
        super().__init__(config)
        self.config = config

        # ESM2 backbone
        self.esm = ESM2(
            num_layers=config.num_layers,
            embed_dim=config.embed_dim,
            attention_heads=config.attention_heads,
            token_dropout=config.token_dropout,
            use_multimer=config.use_multimer,
        )

        # Projection: Linear + ReLU (from Stage 2)
        self.project = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
        )

        # Metadata encoders
        self.assay_emb = nn.Embedding(config.num_assays, config.assay_emb_dim)
        self.temp_linear = nn.Linear(1, config.temp_emb_dim)

        # Feature-conditioned FiLM MLP: [assay_emb; temp_emb; h] -> [gamma, beta]
        meta_dim = config.assay_emb_dim + config.temp_emb_dim
        film_input_dim = meta_dim + config.hidden_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(film_input_dim, config.film_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.film_hidden_dim, 2 * config.hidden_dim),
        )

        # Post-modulation dropout
        self.drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # Readout
        self.readout = nn.Linear(config.hidden_dim, 1)

        self._hidden_dim = config.hidden_dim
        self.post_init()

    def encode(self, chains: torch.Tensor, chain_ids: torch.Tensor) -> torch.Tensor:
        """ESM2 forward + mean pooling over non-special tokens -> [B, embed_dim]."""
        mask = (
            (~chains.eq(self.esm.cls_idx))
            & (~chains.eq(self.esm.eos_idx))
            & (~chains.eq(self.esm.padding_idx))
        )
        chain_out = self.esm(
            chains, chain_ids, repr_layers=[self.config.num_layers]
        )["representations"][self.config.num_layers]

        mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()
        return sum_masked / mask_counts

    def forward(
        self,
        chains: torch.Tensor,
        chain_ids: torch.Tensor,
        assay_idxs: Optional[torch.Tensor] = None,
        temp_floats: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            chains: Token IDs, shape (batch, seq_len).
            chain_ids: Chain membership IDs, shape (batch, seq_len).
                       0 = peptide, 1 = MHC.
            assay_idxs: Assay type indices, shape (batch,).
                        0=SPA, 1=Purified_Fluor, 2=Cellular_Fluor, 3=Other.
                        If None, defaults to 0 (SPA).
            temp_floats: Temperature in Celsius, shape (batch,).
                         If None, defaults to 37.0.
            labels: Optional regression targets, shape (batch,).

        Returns:
            dict with "loss" (if labels provided) and "logits".
        """
        batch_size = chains.shape[0]
        device = chains.device

        # Default assay/temp if not provided
        if assay_idxs is None:
            assay_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)
        if temp_floats is None:
            temp_floats = torch.full(
                (batch_size,), self.config.default_temp_c,
                dtype=torch.float32, device=device,
            )

        # ESM backbone -> mean pool -> project
        r = self.encode(chains, chain_ids)           # [B, embed_dim]
        h = self.project(r)                          # [B, hidden_dim]

        # Metadata encoding
        e_a = self.assay_emb(assay_idxs)             # [B, assay_emb_dim]
        e_t = self.temp_linear(temp_floats.unsqueeze(-1))  # [B, temp_emb_dim]
        u = torch.cat([e_a, e_t, h], dim=-1)         # [B, meta_dim + hidden_dim]

        # FiLM modulation
        film_out = self.film_mlp(u)                   # [B, 2*hidden_dim]
        gamma = film_out[:, :self._hidden_dim]        # [B, hidden_dim]
        beta = film_out[:, self._hidden_dim:]         # [B, hidden_dim]
        h_mod = gamma * h + beta                      # [B, hidden_dim]

        out = self.readout(self.drop(h_mod))          # [B, 1]

        loss = None
        if labels is not None:
            loss = F.mse_loss(out.squeeze(-1), labels)

        return {"loss": loss, "logits": out}
