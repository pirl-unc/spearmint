"""Tests for training scripts: train_binding, train_stability, train_stage3.

Organized bottom-up:
  1. Unit tests — individual functions in isolation
  2. Component tests — classes with minimal wiring
  3. Integration tests — components composed together

Every test is atomic: one assertion per behavior. No try/except.
If a unit test fails, dependent component/integration tests also fail,
making the root cause immediately obvious from the test name.
"""

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from mint_stability.train_binding import (
    MHCBindingDataset,
    MHCBindingWrapper,
    MHCCollateFn,
    classification_metrics,
    load_esm2_config,
    regression_metrics,
    set_seed,
    upgrade_state_dict,
)
from mint_stability.train_stage3 import (
    ASSAY_TO_IDX,
    DEFAULT_TEMP_C,
    Stage3Additive,
    Stage3Calibration,
    Stage3CollateFn,
    Stage3Dataset,
    Stage3FiLM,
    make_loss_fn,
    per_assay_metrics,
)
from mint_stability.backbone import ESM2, Alphabet


# ---------------------------------------------------------------------------
# Helpers — small ESM2 config for fast tests (no checkpoint needed)
# ---------------------------------------------------------------------------

def _small_cfg():
    """Return a tiny ESM2 config namespace for tests."""
    return argparse.Namespace(
        encoder_layers=2,
        encoder_embed_dim=64,
        encoder_attention_heads=4,
        token_dropout=True,
        adam_betas="[0.9,0.98]",
        adam_eps=1e-08,
        weight_decay=0.01,
    )


def _make_base_model(cfg=None, hidden_dim=32, dropout=0.0):
    """Create a small MHCBindingWrapper without loading a checkpoint.

    We bypass the checkpoint-loading constructor by building the components
    directly, matching the MHCBindingWrapper attribute layout.
    """
    if cfg is None:
        cfg = _small_cfg()

    model = nn.Module()
    model.cfg = cfg
    model.sigmoid_output = False
    model.model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        token_dropout=cfg.token_dropout,
        use_multimer=True,
    )
    model.project = nn.Sequential(
        nn.Linear(cfg.encoder_embed_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )

    # Add forward method from MHCBindingWrapper
    def forward(self, chains, chain_ids):
        mask = (
            (~chains.eq(self.model.cls_idx))
            & (~chains.eq(self.model.eos_idx))
            & (~chains.eq(self.model.padding_idx))
        )
        chain_out = self.model(chains, chain_ids, repr_layers=[self.cfg.encoder_layers])[
            "representations"
        ][self.cfg.encoder_layers]
        mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()
        mean_chain_out = sum_masked / mask_counts
        out = self.project(mean_chain_out)
        if self.sigmoid_output:
            out = torch.sigmoid(out)
        return out

    import types
    model.forward = types.MethodType(forward, model)
    model.eval()
    return model


def _make_sample_df(n=5, with_assay=False, with_temp=False):
    """Create a small test DataFrame."""
    df = pd.DataFrame({
        "peptide_sequence": ["AAA", "GGG", "LLL", "VVV", "RRR"][:n],
        "mhc_sequence": ["DDD", "EEE", "FFF", "HHH", "KKK"][:n],
        "label": [0.5, 1.0, 0.3, 0.7, 0.9][:n],
    })
    if with_assay:
        df["assay"] = ["SPA", "Purified_Fluor", "Cellular_Fluor", "Other", "SPA"][:n]
    if with_temp:
        df["temperature_C"] = [37.0, 25.0, 37.0, 50.0, float("nan")][:n]
    return df


def _make_tokens_and_chain_ids(batch_size=2, seq_len=10):
    """Create random token and chain_id tensors for model forward."""
    tokens = torch.randint(4, 30, (batch_size, seq_len))
    chain_ids = torch.zeros(batch_size, seq_len, dtype=torch.int32)
    chain_ids[:, seq_len // 2:] = 1
    return tokens, chain_ids


# ===========================================================================
# Unit Tests — Helper Functions
# ===========================================================================

class TestUpgradeStateDict:
    def test_strips_encoder_prefix(self):
        sd = {"encoder.layers.0.weight": torch.tensor([1.0])}
        result = upgrade_state_dict(sd)
        assert "layers.0.weight" in result
        assert "encoder.layers.0.weight" not in result

    def test_strips_sentence_encoder_prefix(self):
        sd = {"encoder.sentence_encoder.layers.0.weight": torch.tensor([1.0])}
        result = upgrade_state_dict(sd)
        assert "layers.0.weight" in result

    def test_no_prefix_unchanged(self):
        sd = {"layers.0.weight": torch.tensor([1.0])}
        result = upgrade_state_dict(sd)
        assert "layers.0.weight" in result


class TestClassificationMetrics:
    def test_keys(self):
        targets = np.array([0, 1, 1, 0])
        preds = np.array([0.1, 0.9, 0.8, 0.2])
        m = classification_metrics(targets, preds, "test")
        assert set(m.keys()) == {
            "test_Accuracy", "test_AUPRC", "test_F1", "test_AUROC",
        }

    def test_perfect_predictions(self):
        targets = np.array([0, 0, 1, 1])
        preds = np.array([0.0, 0.1, 0.9, 1.0])
        m = classification_metrics(targets, preds, "val")
        assert m["val_Accuracy"] == 1.0
        assert m["val_AUROC"] == 1.0


class TestRegressionMetrics:
    def test_keys(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        m = regression_metrics(targets, preds, "test")
        assert set(m.keys()) == {
            "test_RMSE", "test_Pearson", "test_Spearman",
        }

    def test_perfect_predictions(self):
        targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = regression_metrics(targets, targets.copy(), "val")
        assert m["val_RMSE"] == 0.0
        assert m["val_Pearson"] == pytest.approx(1.0)
        assert m["val_Spearman"] == pytest.approx(1.0)


class TestPerAssayMetrics:
    def test_filters_by_assay(self):
        # 10 SPA, 3 Purified_Fluor (< 5 → skipped)
        targets = np.arange(13, dtype=float)
        preds = targets + 0.1
        assays = np.array([0]*10 + [1]*3)
        m = per_assay_metrics(targets, preds, assays, "val")
        # SPA should be present (10 >= 5)
        assert "val_SPA_RMSE" in m
        # Purified_Fluor should be absent (3 < 5)
        assert "val_Purified_Fluor_RMSE" not in m

    def test_spa_with_enough_samples_has_correlation(self):
        targets = np.arange(10, dtype=float)
        preds = targets * 2 + 1  # perfect linear
        assays = np.zeros(10, dtype=int)
        m = per_assay_metrics(targets, preds, assays, "t")
        assert "t_SPA_Spearman" in m
        assert "t_SPA_Pearson" in m


class TestMakeLossFn:
    def test_mse(self):
        args = argparse.Namespace(loss="mse", huber_delta=1.0)
        fn = make_loss_fn(args)
        assert isinstance(fn, nn.MSELoss)

    def test_huber(self):
        args = argparse.Namespace(loss="huber", huber_delta=0.5)
        fn = make_loss_fn(args)
        assert isinstance(fn, nn.HuberLoss)


# ===========================================================================
# Unit Tests — ESM2 Config
# ===========================================================================

class TestESM2Config:
    def test_bundled_defaults(self):
        """load_esm2_config() with no args loads the bundled esm2_config.json."""
        cfg = load_esm2_config()
        assert cfg.encoder_layers == 33
        assert cfg.encoder_embed_dim == 1280
        assert cfg.encoder_attention_heads == 20
        assert cfg.adam_betas == "[0.9,0.98]"
        assert cfg.adam_eps == 1e-08
        assert cfg.weight_decay == 0.01

    def test_custom_json(self):
        """load_esm2_config(path) loads values from a custom JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"encoder_layers": 6, "encoder_embed_dim": 320}, f)
            f.flush()
            cfg = load_esm2_config(f.name)
        os.unlink(f.name)
        assert cfg.encoder_layers == 6
        assert cfg.encoder_embed_dim == 320

    def test_custom_json_is_standalone(self):
        """Custom JSON replaces (not merges with) bundled defaults."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"weight_decay": 0.1}, f)
            f.flush()
            cfg = load_esm2_config(f.name)
        os.unlink(f.name)
        assert cfg.weight_decay == 0.1
        # Only keys in the JSON file are present
        assert not hasattr(cfg, "encoder_layers")


# ===========================================================================
# Component Tests — Datasets
# ===========================================================================

class TestMHCBindingDataset:
    def test_len(self):
        df = _make_sample_df(n=5)
        ds = MHCBindingDataset(df)
        assert len(ds) == 5

    def test_getitem_3tuple(self):
        df = _make_sample_df(n=3)
        ds = MHCBindingDataset(df)
        item = ds[0]
        assert len(item) == 3
        assert item[0] == "AAA"
        assert item[1] == "DDD"
        assert item[2] == 0.5

class TestStage3Dataset:
    def test_len(self):
        df = _make_sample_df(n=4, with_assay=True, with_temp=True)
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        assert len(ds) == 4

    def test_getitem(self):
        df = _make_sample_df(n=3, with_assay=True, with_temp=True)
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        item = ds[0]
        assert len(item) == 5
        assert item[0] == "AAA"   # peptide
        assert item[1] == "DDD"   # mhc
        assert item[2] == 0.5     # label
        assert item[3] == 0       # SPA index
        assert item[4] == 37.0    # temp as float

    def test_missing_temp(self):
        df = pd.DataFrame({
            "peptide_sequence": ["AAA"],
            "mhc_sequence": ["GGG"],
            "label": [1.0],
            "assay": ["SPA"],
            "temperature_C": [float("nan")],
        })
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        assert ds[0][4] == DEFAULT_TEMP_C

    def test_unknown_assay(self):
        df = pd.DataFrame({
            "peptide_sequence": ["AAA"],
            "mhc_sequence": ["GGG"],
            "label": [1.0],
            "assay": ["WeirdAssay"],
            "temperature_C": [25.0],
        })
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        assert ds[0][3] == ASSAY_TO_IDX["Other"]


# ===========================================================================
# Component Tests — Collate Functions
# ===========================================================================

class TestMHCCollateFn:
    def setup_method(self):
        self.collate = MHCCollateFn()

    def test_output_types(self):
        batch = [("AAA", "GGG", 1.0), ("LL", "RR", 0.0)]
        chains, chain_ids, labels = self.collate(batch)
        assert chains.dtype == torch.int64
        assert chain_ids.dtype == torch.int32
        assert labels.dtype == torch.float32

    def test_output_shapes(self):
        batch = [("AAA", "GGG", 1.0), ("LL", "RR", 0.0)]
        chains, chain_ids, labels = self.collate(batch)
        assert chains.shape[0] == 2
        assert chain_ids.shape == chains.shape
        assert labels.shape == (2,)

    def test_chain_ids(self):
        batch = [("AAA", "GGG", 1.0)]
        chains, chain_ids, labels = self.collate(batch)
        # Peptide: cls + 3 aa + eos = 5 tokens with chain_id=0
        assert (chain_ids[0, :5] == 0).all()
        # MHC: cls + 3 aa + eos = 5 tokens with chain_id=1
        assert (chain_ids[0, 5:] == 1).all()

    def test_padding(self):
        batch = [("AA", "GG", 1.0), ("LLLL", "RRRR", 0.0)]
        chains, chain_ids, _ = self.collate(batch)
        # Shorter sequence should be padded
        padding_idx = self.collate.alphabet.padding_idx
        # Last position of first sample should be padding
        # First sample: 4+4=8 tokens, second: 6+6=12 tokens
        assert chains[0, -1].item() == padding_idx

    def test_j_replacement(self):
        batch_j = [("JAA", "GGG", 1.0)]
        batch_l = [("LAA", "GGG", 1.0)]
        chains_j, _, _ = self.collate(batch_j)
        chains_l, _, _ = self.collate(batch_l)
        assert torch.equal(chains_j, chains_l)

    def test_truncation(self):
        collate = MHCCollateFn(truncation_seq_length=4)
        # Long sequence: cls + 10 aa + eos = 12 tokens, should be truncated to 4
        batch = [("AAAAAAAAAA", "G", 1.0)]
        chains, _, _ = collate(batch)
        # Peptide chain should be at most 4 tokens long
        # Total = truncated_pep + mhc(cls+G+eos=3)
        assert chains.shape[1] <= 4 + 3


class TestStage3CollateFn:
    def setup_method(self):
        self.collate = Stage3CollateFn()

    def test_output_types(self):
        batch = [("AAA", "GGG", 1.0, 0, 37.0), ("LL", "RR", 0.5, 1, 25.0)]
        chains, chain_ids, labels, assay_idxs, temp_floats = self.collate(batch)
        assert chains.dtype == torch.int64
        assert chain_ids.dtype == torch.int32
        assert labels.dtype == torch.float32
        assert assay_idxs.dtype == torch.long
        assert temp_floats.dtype == torch.float32

    def test_output_shapes(self):
        batch = [("AAA", "GGG", 1.0, 0, 37.0), ("LL", "RR", 0.5, 1, 25.0)]
        chains, chain_ids, labels, assay_idxs, temp_floats = self.collate(batch)
        assert chains.shape[0] == 2
        assert assay_idxs.shape == (2,)
        assert temp_floats.shape == (2,)

    def test_temp_is_float(self):
        batch = [("AAA", "GGG", 1.0, 0, 37.0)]
        _, _, _, _, temp_floats = self.collate(batch)
        assert temp_floats[0].item() == 37.0


# ===========================================================================
# Component Tests — Model Wrappers (small ESM2)
# ===========================================================================

class TestMHCBindingWrapperDirect:
    """Test MHCBindingWrapper-like model without needing a checkpoint file."""

    def setup_method(self):
        set_seed(42)
        self.cfg = _small_cfg()
        self.model = _make_base_model(self.cfg, hidden_dim=32)
        self.model.eval()

    def test_forward_shape(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        with torch.no_grad():
            out = self.model(tokens, chain_ids)
        assert out.shape == (2, 1)

    def test_forward_no_metadata(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=1, seq_len=8)
        with torch.no_grad():
            out = self.model(tokens, chain_ids)
        assert out.shape == (1, 1)

    def test_sigmoid_output(self):
        self.model.sigmoid_output = True
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=3, seq_len=10)
        with torch.no_grad():
            out = self.model(tokens, chain_ids)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_deterministic(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        with torch.no_grad():
            out1 = self.model(tokens, chain_ids)
            out2 = self.model(tokens, chain_ids)
        assert torch.equal(out1, out2)


# ===========================================================================
# Component Tests — Stage 3 Models (small ESM2)
# ===========================================================================

class TestStage3FiLM:
    def setup_method(self):
        set_seed(42)
        self.cfg = _small_cfg()
        self.base = _make_base_model(self.cfg, hidden_dim=32)
        self.model = Stage3FiLM(
            self.base,
            num_assays=4,
            assay_emb_dim=8,
            temp_emb_dim=4,
            film_hidden_dim=16,
            dropout=0.0,
        )
        self.model.eval()

    def test_forward_shape(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 1])
        temp_floats = torch.tensor([37.0, 25.0])
        with torch.no_grad():
            out = self.model(tokens, chain_ids, assay_idxs, temp_floats)
        assert out.shape == (2,)

    def test_identity_init(self):
        """At init (gamma=1, beta=0), FiLM output should match base model."""
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 0])
        temp_floats = torch.tensor([37.0, 37.0])
        with torch.no_grad():
            film_out = self.model(tokens, chain_ids, assay_idxs, temp_floats)
            base_out = self.base(tokens, chain_ids).squeeze(-1)
        assert torch.allclose(film_out, base_out, atol=1e-5)

    def test_different_assays(self):
        """Different assay embeddings should produce different outputs."""
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=1, seq_len=10)
        # Perturb assay embeddings so they differ
        with torch.no_grad():
            self.model.assay_emb.weight[0].fill_(1.0)
            self.model.assay_emb.weight[1].fill_(-1.0)
            # Break FiLM identity init so embeddings actually affect output
            self.model.film_mlp[-1].weight.normal_(0, 0.01)
        self.model.eval()
        with torch.no_grad():
            out_0 = self.model(tokens, chain_ids, torch.tensor([0]), torch.tensor([37.0]))
            out_1 = self.model(tokens, chain_ids, torch.tensor([1]), torch.tensor([37.0]))
        assert not torch.equal(out_0, out_1)

    def test_encode_shape(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        with torch.no_grad():
            r = self.model.encode(tokens, chain_ids)
        assert r.shape == (2, self.cfg.encoder_embed_dim)

    def test_train_mode_esm_eval(self):
        self.model.train()
        assert not self.model.esm.training


class TestStage3Calibration:
    def setup_method(self):
        set_seed(42)
        self.cfg = _small_cfg()
        self.base = _make_base_model(self.cfg, hidden_dim=32)
        self.model = Stage3Calibration(
            self.base,
            num_assays=4,
            assay_emb_dim=8,
            temp_emb_dim=4,
            residual_hidden_dim=16,
        )
        self.model.eval()

    def test_forward_shape(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 1])
        temp_floats = torch.tensor([37.0, 25.0])
        with torch.no_grad():
            out = self.model(tokens, chain_ids, assay_idxs, temp_floats)
        assert out.shape == (2,)

    def test_identity_init(self):
        """At init (a=1, b=0, delta=0), output should match base model."""
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 0])
        temp_floats = torch.tensor([37.0, 37.0])
        with torch.no_grad():
            cal_out = self.model(tokens, chain_ids, assay_idxs, temp_floats)
            base_out = self.base(tokens, chain_ids).squeeze(-1)
        assert torch.allclose(cal_out, base_out, atol=1e-5)

    def test_different_assays(self):
        """Per-assay scale/bias should make different assays produce different outputs after training."""
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=1, seq_len=10)
        # Modify calib params to make assays differ
        with torch.no_grad():
            self.model.calib_scale[0] = 2.0
            self.model.calib_bias[1] = 5.0
        self.model.eval()
        with torch.no_grad():
            out_0 = self.model(tokens, chain_ids, torch.tensor([0]), torch.tensor([37.0]))
            out_1 = self.model(tokens, chain_ids, torch.tensor([1]), torch.tensor([37.0]))
        assert not torch.equal(out_0, out_1)


class TestStage3Additive:
    def setup_method(self):
        set_seed(42)
        self.cfg = _small_cfg()
        self.base = _make_base_model(self.cfg, hidden_dim=32)
        self.model = Stage3Additive(
            self.base,
            num_assays=4,
            assay_emb_dim=8,
            temp_emb_dim=4,
            dropout=0.0,
        )
        self.model.eval()

    def test_forward_shape(self):
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 1])
        temp_floats = torch.tensor([37.0, 25.0])
        with torch.no_grad():
            out = self.model(tokens, chain_ids, assay_idxs, temp_floats)
        assert out.shape == (2,)

    def test_identity_init(self):
        """At init (residual=0), output should match base model."""
        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 0])
        temp_floats = torch.tensor([37.0, 37.0])
        with torch.no_grad():
            add_out = self.model(tokens, chain_ids, assay_idxs, temp_floats)
            base_out = self.base(tokens, chain_ids).squeeze(-1)
        assert torch.allclose(add_out, base_out, atol=1e-5)

    def test_train_mode_esm_eval(self):
        self.model.train()
        assert not self.model.esm.training


# ===========================================================================
# Integration Tests — Dataset → Collate → Model
# ===========================================================================

class TestBindingPipeline:
    def test_full_pipeline(self):
        """MHCBindingDataset → MHCCollateFn → base model forward → correct shape."""
        set_seed(42)
        df = _make_sample_df(n=3)
        ds = MHCBindingDataset(df)
        collate = MHCCollateFn()
        batch = collate([ds[i] for i in range(3)])
        chains, chain_ids, labels = batch

        model = _make_base_model()
        model.eval()
        with torch.no_grad():
            out = model(chains, chain_ids)
        assert out.shape == (3, 1)
        assert labels.shape == (3,)


class TestStage3PipelineFiLM:
    def test_full_pipeline(self):
        """Stage3Dataset → Stage3CollateFn → Stage3FiLM forward → correct shape."""
        set_seed(42)
        df = _make_sample_df(n=3, with_assay=True, with_temp=True)
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        collate = Stage3CollateFn()
        batch = collate([ds[i] for i in range(3)])
        chains, chain_ids, labels, assay_idxs, temp_floats = batch

        base = _make_base_model(hidden_dim=32)
        model = Stage3FiLM(base, num_assays=4, assay_emb_dim=8, temp_emb_dim=4,
                           film_hidden_dim=16)
        model.eval()
        with torch.no_grad():
            out = model(chains, chain_ids, assay_idxs, temp_floats)
        assert out.shape == (3,)


class TestStage3PipelineCalibration:
    def test_full_pipeline(self):
        """Stage3Dataset → Stage3CollateFn → Stage3Calibration forward → correct shape."""
        set_seed(42)
        df = _make_sample_df(n=3, with_assay=True, with_temp=True)
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        collate = Stage3CollateFn()
        batch = collate([ds[i] for i in range(3)])
        chains, chain_ids, labels, assay_idxs, temp_floats = batch

        base = _make_base_model(hidden_dim=32)
        model = Stage3Calibration(base, num_assays=4, assay_emb_dim=8, temp_emb_dim=4,
                                  residual_hidden_dim=16)
        model.eval()
        with torch.no_grad():
            out = model(chains, chain_ids, assay_idxs, temp_floats)
        assert out.shape == (3,)


class TestStage3PipelineAdditive:
    def test_full_pipeline(self):
        """Stage3Dataset → Stage3CollateFn → Stage3Additive forward → correct shape."""
        set_seed(42)
        df = _make_sample_df(n=3, with_assay=True, with_temp=True)
        ds = Stage3Dataset(df, assay_col="assay", temp_col="temperature_C")
        collate = Stage3CollateFn()
        batch = collate([ds[i] for i in range(3)])
        chains, chain_ids, labels, assay_idxs, temp_floats = batch

        base = _make_base_model(hidden_dim=32)
        model = Stage3Additive(base, num_assays=4, assay_emb_dim=8, temp_emb_dim=4)
        model.eval()
        with torch.no_grad():
            out = model(chains, chain_ids, assay_idxs, temp_floats)
        assert out.shape == (3,)


# ===========================================================================
# Integration Tests — Checkpoint Save/Load
# ===========================================================================

class TestCheckpointRoundTrip:
    def test_base_model_save_load(self):
        """Save base model state_dict → load into new model → same output."""
        set_seed(42)
        model1 = _make_base_model(hidden_dim=32)
        model1.eval()

        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)

        with torch.no_grad():
            out1 = model1(tokens, chain_ids)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": model1.state_dict()}, f.name)
            ckpt_path = f.name

        set_seed(42)
        model2 = _make_base_model(hidden_dim=32)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()
        os.unlink(ckpt_path)

        with torch.no_grad():
            out2 = model2(tokens, chain_ids)
        assert torch.equal(out1, out2)

    def test_film_save_load(self):
        """Save FiLM state_dict → load into new model → same output."""
        set_seed(42)
        base1 = _make_base_model(hidden_dim=32)
        model1 = Stage3FiLM(base1, num_assays=4, assay_emb_dim=8, temp_emb_dim=4,
                            film_hidden_dim=16)
        model1.eval()

        tokens, chain_ids = _make_tokens_and_chain_ids(batch_size=2, seq_len=10)
        assay_idxs = torch.tensor([0, 1])
        temp_floats = torch.tensor([37.0, 25.0])

        with torch.no_grad():
            out1 = model1(tokens, chain_ids, assay_idxs, temp_floats)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"model_state_dict": model1.state_dict()}, f.name)
            ckpt_path = f.name

        set_seed(42)
        base2 = _make_base_model(hidden_dim=32)
        model2 = Stage3FiLM(base2, num_assays=4, assay_emb_dim=8, temp_emb_dim=4,
                            film_hidden_dim=16)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state_dict"])
        model2.eval()
        os.unlink(ckpt_path)

        with torch.no_grad():
            out2 = model2(tokens, chain_ids, assay_idxs, temp_floats)
        assert torch.equal(out1, out2)


# ===========================================================================
# Seed Tests
# ===========================================================================

class TestSeeding:
    def test_seed_reproducibility(self):
        """Two model inits with the same seed produce identical parameters."""
        set_seed(123)
        m1 = _make_base_model(hidden_dim=32)

        set_seed(123)
        m2 = _make_base_model(hidden_dim=32)

        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)
