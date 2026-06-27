"""Tests for model instantiation and forward pass."""

import torch
import pytest

from mint_stability import (
    MintStabilityConfig,
    SpearmintConfig,
    MintStabilityForRegression,
    SpearmintForStabilityPrediction,
    MintTokenizer,
    SpearmintTokenizer,
)


class TestMintStabilityForRegression:
    def setup_method(self):
        self.config = MintStabilityConfig(
            num_layers=2,  # Small for fast tests
            embed_dim=64,
            attention_heads=4,
            hidden_dim=32,
            dropout=0.1,
            output_size=1,
            use_multimer=True,
        )
        self.model = MintStabilityForRegression(self.config)
        self.model.eval()
        self.tokenizer = MintTokenizer()

    def test_forward_output_shape(self):
        chains, chain_ids = self.tokenizer.prepare_input("AAA", "GGG")
        with torch.no_grad():
            out = self.model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
        assert "logits" in out
        assert out["logits"].shape == (1, 1)
        assert out["loss"] is None

    def test_forward_with_labels(self):
        chains, chain_ids = self.tokenizer.prepare_input("AAA", "GGG")
        labels = torch.tensor([0.5])
        with torch.no_grad():
            out = self.model(chains.unsqueeze(0), chain_ids.unsqueeze(0), labels=labels)
        assert out["loss"] is not None
        assert out["loss"].shape == ()

    def test_forward_batch(self):
        chains, chain_ids = self.tokenizer.prepare_batch(
            ["AAA", "GGGGG"], ["RRR", "DDDDD"]
        )
        with torch.no_grad():
            out = self.model(chains, chain_ids)
        assert out["logits"].shape == (2, 1)

    def test_sigmoid_output(self):
        config = MintStabilityConfig(
            num_layers=2, embed_dim=64, attention_heads=4,
            hidden_dim=32, sigmoid_output=True,
        )
        model = MintStabilityForRegression(config)
        model.eval()
        chains, chain_ids = self.tokenizer.prepare_input("AAA", "GGG")
        with torch.no_grad():
            out = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
        # Sigmoid output should be in [0, 1]
        assert 0 <= out["logits"].item() <= 1

    def test_deterministic(self):
        """Same input should produce same output."""
        chains, chain_ids = self.tokenizer.prepare_input("GILGFVFTL", "MAVMAPRTL")
        with torch.no_grad():
            out1 = self.model(chains.unsqueeze(0), chain_ids.unsqueeze(0))["logits"]
            out2 = self.model(chains.unsqueeze(0), chain_ids.unsqueeze(0))["logits"]
        assert torch.equal(out1, out2)


class TestSpearmintForStabilityPrediction:
    def setup_method(self):
        self.config = SpearmintConfig(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            hidden_dim=32,
            num_assays=4,
            assay_emb_dim=8,
            temp_emb_dim=4,
            film_hidden_dim=32,
            dropout=0.0,
            use_multimer=True,
        )
        self.model = SpearmintForStabilityPrediction(self.config)
        self.model.eval()
        self.tokenizer = SpearmintTokenizer()

    def test_forward_output_shape(self):
        chains, chain_ids, assay_idx, temp = self.tokenizer.prepare_input(
            "AAA", "GGG", assay="SPA", temperature_c=37.0
        )
        with torch.no_grad():
            out = self.model(chains.unsqueeze(0), chain_ids.unsqueeze(0), assay_idx, temp)
        assert out["logits"].shape == (1, 1)
        assert out["loss"] is None

    def test_forward_defaults(self):
        """Forward should work without assay/temp (uses defaults)."""
        chains, chain_ids, _, _ = self.tokenizer.prepare_input("AAA", "GGG")
        with torch.no_grad():
            out = self.model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
        assert out["logits"].shape == (1, 1)

    def test_forward_with_labels(self):
        chains, chain_ids, assay_idx, temp = self.tokenizer.prepare_input("AAA", "GGG")
        labels = torch.tensor([1.5])
        with torch.no_grad():
            out = self.model(
                chains.unsqueeze(0), chain_ids.unsqueeze(0),
                assay_idx, temp, labels=labels,
            )
        assert out["loss"] is not None

    def test_different_assays_produce_different_outputs(self):
        """FiLM conditioning should make different assays produce different outputs."""
        chains, chain_ids, _, _ = self.tokenizer.prepare_input("AAA", "GGG")
        chains_b = chains.unsqueeze(0)
        cids_b = chain_ids.unsqueeze(0)

        with torch.no_grad():
            out_spa = self.model(
                chains_b, cids_b,
                torch.tensor([0]), torch.tensor([37.0]),
            )["logits"]
            out_fluor = self.model(
                chains_b, cids_b,
                torch.tensor([1]), torch.tensor([37.0]),
            )["logits"]
        # Different assays should give different predictions
        assert not torch.equal(out_spa, out_fluor)

    def test_encode_method(self):
        """encode() should return mean-pooled representations."""
        chains, chain_ids, _, _ = self.tokenizer.prepare_input("AAA", "GGG")
        with torch.no_grad():
            r = self.model.encode(chains.unsqueeze(0), chain_ids.unsqueeze(0))
        assert r.shape == (1, 64)  # embed_dim=64
