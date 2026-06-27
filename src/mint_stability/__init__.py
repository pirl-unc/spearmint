"""MINT pMHC-I stability prediction models.

Provides three model stages sharing a common ESM2 backbone:
  - S1 (MintStabilityForRegression): Binding affinity prediction
  - S2 (MintStabilityForRegression): Stability (half-life) prediction
  - S3 (SpearmintForStabilityPrediction): Assay-aware stability with FiLM conditioning

Usage:
    from mint_stability import MintStabilityForRegression, MintTokenizer

    model = MintStabilityForRegression.from_pretrained("dkarthikeyan1/mint-2stage-stability")
    tokenizer = MintTokenizer()

    chains, chain_ids = tokenizer.prepare_input("GILGFVFTL", mhc_sequence)
    output = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
"""

from .configuration_mint import MintStabilityConfig
from .configuration_spearmint import SpearmintConfig
from .modeling_base import StabilityPreTrainedModel
from .modeling_mint import MintStabilityForRegression, MintStabilityPreTrainedModel
from .modeling_spearmint import (
    SpearmintForStabilityPrediction,
    SpearmintPreTrainedModel,
)
from .tokenizer import MintTokenizer, SpearmintTokenizer

__all__ = [
    # Inference models
    "MintStabilityConfig",
    "SpearmintConfig",
    "StabilityPreTrainedModel",
    "MintStabilityForRegression",
    "MintStabilityPreTrainedModel",
    "SpearmintForStabilityPrediction",
    "SpearmintPreTrainedModel",
    "MintTokenizer",
    "SpearmintTokenizer",
]
