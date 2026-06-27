"""Tests for build_hf.py — monolithic HF file generation."""

import importlib
import os
import sys
import tempfile

import torch
import pytest

from mint_stability.build_hf import _build_monolithic, _extract_class


class TestExtractClass:
    def test_extract_simple_class(self):
        source = '''
class Foo:
    def bar(self):
        pass

class Baz:
    pass
'''
        result = _extract_class(source, "Foo")
        assert "class Foo:" in result
        assert "def bar" in result
        assert "class Baz" not in result

    def test_extract_missing_class_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _extract_class("class Foo: pass", "Bar")


class TestBuildMonolithic:
    @pytest.mark.parametrize("stage", ["s1", "s2"])
    def test_s2_generates_valid_python(self, stage):
        """Generated S2 file should be valid Python that can be compiled."""
        modeling_src, config_src, config_fn, modeling_fn = _build_monolithic(stage)
        # Should not raise
        compile(modeling_src, modeling_fn, "exec")
        compile(config_src, config_fn, "exec")

    def test_s3_generates_valid_python(self):
        """Generated S3 file should be valid Python that can be compiled."""
        modeling_src, config_src, config_fn, modeling_fn = _build_monolithic("s3")
        compile(modeling_src, modeling_fn, "exec")
        compile(config_src, config_fn, "exec")

    def test_s2_contains_expected_classes(self):
        modeling_src, _, _, _ = _build_monolithic("s2")
        assert "class MintStabilityForRegression" in modeling_src
        assert "class MintTokenizer" in modeling_src
        assert "class ESM2" in modeling_src
        assert "class Alphabet" in modeling_src
        # Should NOT contain S3 classes
        assert "class SpearmintForStabilityPrediction" not in modeling_src
        assert "class SpearmintTokenizer" not in modeling_src

    def test_s3_contains_expected_classes(self):
        modeling_src, _, _, _ = _build_monolithic("s3")
        assert "class SpearmintForStabilityPrediction" in modeling_src
        assert "class SpearmintTokenizer" in modeling_src
        assert "class ESM2" in modeling_src
        # Should NOT contain S2 classes
        assert "class MintStabilityForRegression" not in modeling_src
        assert "class MintTokenizer" not in modeling_src

    def test_config_import_fallback(self):
        """Config import fallback must NOT be a bare top-level
        `from <config_module> import ...` -- transformers' check_imports would flag
        it as a missing pip dependency and break trust_remote_code loading. The
        fallback uses importlib instead (see test_*_trust_remote_code_round_trip)."""
        modeling_src, _, _, _ = _build_monolithic("s2")
        assert "try:" in modeling_src
        assert "from .configuration_mint import MintStabilityConfig" in modeling_src
        assert 'import_module("configuration_mint")' in modeling_src
        assert "from configuration_mint import" not in modeling_src

    def test_s2_importable_and_runnable(self):
        """Generated S2 file should be importable and produce model output."""
        modeling_src, config_src, config_fn, modeling_fn = _build_monolithic("s2")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write files
            with open(os.path.join(tmpdir, modeling_fn), "w") as f:
                f.write(modeling_src)
            with open(os.path.join(tmpdir, config_fn), "w") as f:
                f.write(config_src)

            # Import
            sys.path.insert(0, tmpdir)
            try:
                config_mod = importlib.import_module(config_fn.replace(".py", ""))
                modeling_mod = importlib.import_module(modeling_fn.replace(".py", ""))

                config = config_mod.MintStabilityConfig(
                    num_layers=2, embed_dim=64, attention_heads=4, hidden_dim=32,
                )
                model = modeling_mod.MintStabilityForRegression(config)
                model.eval()

                tokenizer = modeling_mod.MintTokenizer()
                chains, chain_ids = tokenizer.prepare_input("AAA", "GGG")
                with torch.no_grad():
                    out = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
                assert out["logits"].shape == (1, 1)
            finally:
                sys.path.pop(0)
                for mod_name in [config_fn.replace(".py", ""), modeling_fn.replace(".py", "")]:
                    sys.modules.pop(mod_name, None)

    def test_s3_importable_and_runnable(self):
        """Generated S3 file should be importable and produce model output."""
        modeling_src, config_src, config_fn, modeling_fn = _build_monolithic("s3")

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, modeling_fn), "w") as f:
                f.write(modeling_src)
            with open(os.path.join(tmpdir, config_fn), "w") as f:
                f.write(config_src)

            sys.path.insert(0, tmpdir)
            try:
                config_mod = importlib.import_module(config_fn.replace(".py", ""))
                modeling_mod = importlib.import_module(modeling_fn.replace(".py", ""))

                config = config_mod.SpearmintConfig(
                    num_layers=2, embed_dim=64, attention_heads=4,
                    hidden_dim=32, num_assays=4, assay_emb_dim=8,
                    temp_emb_dim=4, film_hidden_dim=32,
                )
                model = modeling_mod.SpearmintForStabilityPrediction(config)
                model.eval()

                tokenizer = modeling_mod.SpearmintTokenizer()
                chains, chain_ids, assay_idx, temp = tokenizer.prepare_input(
                    "AAA", "GGG", assay="SPA", temperature_c=37.0
                )
                with torch.no_grad():
                    out = model(
                        chains.unsqueeze(0), chain_ids.unsqueeze(0),
                        assay_idx, temp,
                    )
                assert out["logits"].shape == (1, 1)
            finally:
                sys.path.pop(0)
                for mod_name in [config_fn.replace(".py", ""), modeling_fn.replace(".py", "")]:
                    sys.modules.pop(mod_name, None)

    def test_s1_importable_and_runnable(self):
        """Generated S1 file should be importable and produce model output."""
        modeling_src, config_src, config_fn, modeling_fn = _build_monolithic("s1")

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, modeling_fn), "w") as f:
                f.write(modeling_src)
            with open(os.path.join(tmpdir, config_fn), "w") as f:
                f.write(config_src)

            sys.path.insert(0, tmpdir)
            try:
                config_mod = importlib.import_module(config_fn.replace(".py", ""))
                modeling_mod = importlib.import_module(modeling_fn.replace(".py", ""))

                config = config_mod.MintStabilityConfig(
                    num_layers=2, embed_dim=64, attention_heads=4, hidden_dim=32,
                )
                model = modeling_mod.MintStabilityForRegression(config)
                model.eval()

                tokenizer = modeling_mod.MintTokenizer()
                chains, chain_ids = tokenizer.prepare_input("AAA", "GGG")
                with torch.no_grad():
                    out = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
                assert out["logits"].shape == (1, 1)
            finally:
                sys.path.pop(0)
                for mod_name in [config_fn.replace(".py", ""), modeling_fn.replace(".py", "")]:
                    sys.modules.pop(mod_name, None)

    def test_s3_trust_remote_code_round_trip(self):
        """Generated files must load via the REAL HuggingFace
        AutoModel(trust_remote_code=True) + dynamic-tokenizer path, not just a
        standalone import. Guards the config-import fallback against
        transformers' check_imports breaking remote loading."""
        from transformers import AutoModel
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from mint_stability import SpearmintConfig, SpearmintForStabilityPrediction

        modeling_src, config_src, config_fn, modeling_fn = _build_monolithic("s3")
        config = SpearmintConfig(
            num_layers=2, embed_dim=64, attention_heads=4, hidden_dim=32,
            num_assays=4, assay_emb_dim=8, temp_emb_dim=4, film_hidden_dim=16,
        )
        config.auto_map = {
            "AutoConfig": "configuration_spearmint.SpearmintConfig",
            "AutoModel": "modeling_spearmint.SpearmintForStabilityPrediction",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SpearmintForStabilityPrediction(config).eval()
            model.config.auto_map = config.auto_map
            model.save_pretrained(tmpdir, safe_serialization=True)
            with open(os.path.join(tmpdir, modeling_fn), "w") as f:
                f.write(modeling_src)
            with open(os.path.join(tmpdir, config_fn), "w") as f:
                f.write(config_src)

            loaded = AutoModel.from_pretrained(tmpdir, trust_remote_code=True)
            loaded.eval()
            TokCls = get_class_from_dynamic_module(
                "modeling_spearmint.SpearmintTokenizer", tmpdir,
                trust_remote_code=True,
            )
            tokenizer = TokCls()
            chains, chain_ids, assay_idx, temp = tokenizer.prepare_input(
                "GILGFVFTL", "MAVMAPRTLAA", assay="SPA", temperature_c=37.0,
            )
            with torch.no_grad():
                out = loaded(chains.unsqueeze(0), chain_ids.unsqueeze(0), assay_idx, temp)
            assert out["logits"].shape == (1, 1)

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            _build_monolithic("s4")
