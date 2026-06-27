"""MINT Stability model (S1 binding affinity / S2 stability)."""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from .backbone import ESM2
from .configuration_mint import MintStabilityConfig
from .modeling_base import StabilityPreTrainedModel

__all__ = ["MintStabilityPreTrainedModel", "MintStabilityForRegression"]


class MintStabilityPreTrainedModel(StabilityPreTrainedModel):
    """Base class for MINT stability models."""

    config_class = MintStabilityConfig


class MintStabilityForRegression(MintStabilityPreTrainedModel):
    """MINT backbone + projection head for pMHC stability prediction.

    Architecture: ESM2-650M with optional cross-chain multimer attention,
    mean-pooled over non-special tokens, followed by a projection head
    (Linear -> ReLU -> Dropout -> Linear) outputting a scalar prediction
    (half-life in hours).
    """

    config_class = MintStabilityConfig
    # lm_head.weight is tied to embed_tokens.weight in ESM2 but is never
    # used during stability inference -- suppress the missing-key warning.
    _keys_to_ignore_on_load_missing = ["model.lm_head.weight"]

    def __init__(self, config: MintStabilityConfig):
        super().__init__(config)
        self.config = config

        self.model = ESM2(
            num_layers=config.num_layers,
            embed_dim=config.embed_dim,
            attention_heads=config.attention_heads,
            token_dropout=config.token_dropout,
            use_multimer=config.use_multimer,
        )

        self.project = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_size),
        )

        self.sigmoid_output = config.sigmoid_output
        self.post_init()

    def forward(
        self,
        chains: torch.Tensor,
        chain_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            chains: Token IDs, shape (batch, seq_len).
            chain_ids: Chain membership IDs, shape (batch, seq_len).
                       0 = peptide, 1 = MHC.
            labels: Optional regression targets, shape (batch,).

        Returns:
            dict with "loss" (if labels provided) and "logits".
        """
        mask = (
            (~chains.eq(self.model.cls_idx))
            & (~chains.eq(self.model.eos_idx))
            & (~chains.eq(self.model.padding_idx))
        )
        chain_out = self.model(
            chains, chain_ids, repr_layers=[self.config.num_layers]
        )["representations"][self.config.num_layers]

        mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()
        mean_chain_out = sum_masked / mask_counts

        out = self.project(mean_chain_out)
        if self.sigmoid_output:
            out = torch.sigmoid(out)

        loss = None
        if labels is not None:
            loss = F.mse_loss(out.squeeze(-1), labels)

        return {"loss": loss, "logits": out}
