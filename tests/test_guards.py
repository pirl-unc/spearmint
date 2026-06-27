"""Tests for fail-closed guards: _load_checked and from_pretrained error paths."""

import os
import tempfile

import torch
import torch.nn as nn
import pytest

from mint_stability.convert_checkpoint import _load_checked
from mint_stability import (
    MintStabilityConfig,
    MintStabilityForRegression,
    SpearmintConfig,
    SpearmintForStabilityPrediction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_mint_config():
    return MintStabilityConfig(
        num_layers=2, embed_dim=64, attention_heads=4,
        hidden_dim=32, dropout=0.1, output_size=1, use_multimer=True,
    )


def _tiny_spearmint_config():
    return SpearmintConfig(
        num_layers=2, embed_dim=64, attention_heads=4,
        hidden_dim=32, num_assays=4, assay_emb_dim=8,
        temp_emb_dim=4, film_hidden_dim=32, dropout=0.0,
        use_multimer=True,
    )


def _clone_state_dict(sd):
    """Clone all tensors to break shared-memory ties (e.g. ESM2 tied lm_head)."""
    return {k: v.clone() for k, v in sd.items()}


def _save_state_dict_to_dir(sd, config, tmpdir):
    """Save a state_dict + config to a directory, handling tied weights."""
    sd = _clone_state_dict(sd)
    try:
        from safetensors.torch import save_file
        save_file(sd, os.path.join(tmpdir, "model.safetensors"))
    except ImportError:
        torch.save(sd, os.path.join(tmpdir, "pytorch_model.bin"))
    config.save_pretrained(tmpdir)


def _save_model_to_dir(model, config, tmpdir):
    """Save model weights + config to a directory."""
    _save_state_dict_to_dir(model.state_dict(), config, tmpdir)


# ---------------------------------------------------------------------------
# TestLoadChecked — unit tests for _load_checked
# ---------------------------------------------------------------------------

class TestLoadChecked:
    def setup_method(self):
        self.model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def test_perfect_match(self):
        """All keys align — no error."""
        sd = self.model.state_dict()
        _load_checked(self.model, sd)

    def test_unexpected_keys_raises(self):
        """Extra key in state_dict should raise RuntimeError."""
        sd = self.model.state_dict()
        sd["bogus_param"] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="Unexpected keys"):
            _load_checked(self.model, sd)

    def test_missing_keys_raises(self):
        """Key absent from state_dict should raise RuntimeError."""
        sd = self.model.state_dict()
        del sd["0.weight"]
        with pytest.raises(RuntimeError, match="Missing keys"):
            _load_checked(self.model, sd)

    def test_allowed_missing_skipped(self):
        """Key in allowed_missing should not cause an error."""
        sd = self.model.state_dict()
        del sd["0.weight"]
        # Should not raise
        _load_checked(self.model, sd, allowed_missing={"0.weight"})

    def test_allowed_missing_plus_unexpected_still_raises(self):
        """allowed_missing does not suppress unexpected keys."""
        sd = self.model.state_dict()
        del sd["0.weight"]
        sd["bogus_param"] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="Unexpected keys"):
            _load_checked(self.model, sd, allowed_missing={"0.weight"})


# ---------------------------------------------------------------------------
# TestFromPretrainedGuards — integration tests using saved models
# ---------------------------------------------------------------------------

class TestFromPretrainedGuards:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.tmp_path = tmp_path
        self.config = _tiny_mint_config()
        self.model = MintStabilityForRegression(self.config)
        self.model.eval()

    def test_round_trip_identical_output(self):
        """save -> from_pretrained -> same forward output."""
        _save_model_to_dir(self.model, self.config, str(self.tmp_path))

        loaded = MintStabilityForRegression.from_pretrained(str(self.tmp_path))
        loaded.eval()

        from mint_stability import MintTokenizer
        tokenizer = MintTokenizer()
        chains, chain_ids = tokenizer.prepare_input("AAA", "GGG")
        chains = chains.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        with torch.no_grad():
            orig_out = self.model(chains, chain_ids)["logits"]
            loaded_out = loaded(chains, chain_ids)["logits"]

        assert torch.allclose(orig_out, loaded_out, atol=1e-6)

    def test_unexpected_keys_raises(self):
        """Injecting an extra key into the saved weights should raise."""
        sd = self.model.state_dict()
        sd["bogus_extra_key"] = torch.zeros(3)
        _save_state_dict_to_dir(sd, self.config, str(self.tmp_path))

        with pytest.raises(RuntimeError, match="Unexpected keys"):
            MintStabilityForRegression.from_pretrained(str(self.tmp_path))

    def test_missing_keys_raises(self):
        """Removing a required key from saved weights should raise."""
        sd = self.model.state_dict()
        # Remove a key that isn't in _keys_to_ignore_on_load_missing
        del sd["project.0.weight"]
        _save_state_dict_to_dir(sd, self.config, str(self.tmp_path))

        with pytest.raises(RuntimeError, match="Missing keys"):
            MintStabilityForRegression.from_pretrained(str(self.tmp_path))

    def test_no_weights_file_raises(self):
        """Empty dir (config only, no weights) should raise FileNotFoundError."""
        self.config.save_pretrained(str(self.tmp_path))
        # Don't save any weights file
        with pytest.raises(FileNotFoundError, match="No model weights"):
            MintStabilityForRegression.from_pretrained(str(self.tmp_path))


# ---------------------------------------------------------------------------
# TestLoadTransfer — load_transfer allows missing keys, rejects unexpected (P21)
# ---------------------------------------------------------------------------

from mint_stability._compat import load_transfer, torch_load


class TestLoadTransfer:
    def setup_method(self):
        self.model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

    def test_same_arch_no_raise_empty_missing(self):
        """Identical architecture transfers cleanly with no missing keys."""
        missing = load_transfer(self.model, self.model.state_dict(), label="same")
        assert list(missing) == []

    def test_missing_keys_allowed_and_reported(self):
        """Missing keys (e.g. a re-initialized head) are allowed and returned, not raised."""
        sd = self.model.state_dict()
        del sd["2.weight"]
        missing = load_transfer(self.model, sd, label="reinit-head")
        assert "2.weight" in missing

    def test_unexpected_keys_raise(self):
        """An unexpected key signals a checkpoint mismatch and must raise."""
        sd = self.model.state_dict()
        sd["stray_param"] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="Unexpected keys"):
            load_transfer(self.model, sd, label="bad")

    def test_label_surfaced_in_error(self):
        """The label is included in the error message for debuggability."""
        sd = self.model.state_dict()
        sd["stray_param"] = torch.zeros(3)
        with pytest.raises(RuntimeError, match="transfer"):
            load_transfer(self.model, sd, label="S1->S2 transfer")


class TestTorchLoadCompat:
    def test_round_trip_with_weights_only(self):
        """torch_load accepts explicit weights_only= and round-trips a state_dict."""
        sd = nn.Linear(3, 3).state_dict()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "w.pt")
            torch.save(sd, p)
            loaded = torch_load(p, map_location="cpu", weights_only=True)
        assert set(loaded.keys()) == set(sd.keys())
