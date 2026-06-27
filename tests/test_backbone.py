"""Tests for the ESM2 backbone components."""

import torch
import pytest

from mint_stability.backbone import Alphabet, ESM2, RotaryEmbedding


class TestAlphabet:
    def setup_method(self):
        self.alphabet = Alphabet.from_architecture("ESM-1b")

    def test_special_tokens(self):
        assert self.alphabet.cls_idx is not None
        assert self.alphabet.eos_idx is not None
        assert self.alphabet.padding_idx is not None
        assert self.alphabet.mask_idx is not None

    def test_encode_basic(self):
        encoded = self.alphabet.encode("<cls> A G <eos>")
        assert len(encoded) == 4
        assert encoded[0] == self.alphabet.cls_idx
        assert encoded[-1] == self.alphabet.eos_idx

    def test_all_amino_acids_have_indices(self):
        standard_aa = "LAGVSERTIDPKQNFYMHWC"
        for aa in standard_aa:
            idx = self.alphabet.get_idx(aa)
            assert idx != self.alphabet.unk_idx, f"Amino acid {aa} mapped to unk"

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError):
            Alphabet.from_architecture("nonexistent")

    def test_len(self):
        assert len(self.alphabet) == len(self.alphabet.all_toks)


class TestESM2:
    def setup_method(self):
        self.model = ESM2(
            num_layers=2,
            embed_dim=64,
            attention_heads=4,
            token_dropout=True,
            use_multimer=True,
        )
        self.model.eval()

    def test_forward_output_keys(self):
        tokens = torch.randint(4, 30, (1, 10))
        chain_ids = torch.zeros(1, 10, dtype=torch.long)
        out = self.model(tokens, chain_ids, repr_layers=[2])
        assert "logits" in out
        assert "representations" in out
        assert 2 in out["representations"]

    def test_repr_layers(self):
        tokens = torch.randint(4, 30, (1, 10))
        out = self.model(tokens, repr_layers=[0, 1, 2])
        assert 0 in out["representations"]
        assert 1 in out["representations"]
        assert 2 in out["representations"]

    def test_repr_shape(self):
        tokens = torch.randint(4, 30, (1, 10))
        out = self.model(tokens, repr_layers=[2])
        assert out["representations"][2].shape == (1, 10, 64)

    def test_multimer_attention_uses_chain_ids(self):
        """With multimer attention, different chain_ids should produce different outputs."""
        tokens = torch.randint(4, 30, (1, 10))
        # All same chain
        chain_ids_same = torch.zeros(1, 10, dtype=torch.long)
        # Two chains
        chain_ids_diff = torch.cat([
            torch.zeros(1, 5, dtype=torch.long),
            torch.ones(1, 5, dtype=torch.long),
        ], dim=1)

        with torch.no_grad():
            out_same = self.model(tokens, chain_ids_same, repr_layers=[2])
            out_diff = self.model(tokens, chain_ids_diff, repr_layers=[2])

        # Outputs should differ when chain assignments change
        assert not torch.equal(
            out_same["representations"][2],
            out_diff["representations"][2],
        )


class TestRotaryEmbedding:
    def test_output_shapes(self):
        rot = RotaryEmbedding(dim=32)
        q = torch.randn(2, 10, 32)
        k = torch.randn(2, 10, 32)
        q_rot, k_rot = rot(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


class TestRotaryEmbeddingDtype:
    def test_dtype_switch_refreshes_cache(self):
        """Cache must be keyed on dtype (P30): reusing one seq_len across dtypes
        must not return a stale fp32 cos/sin table for an fp16 input."""
        rot = RotaryEmbedding(dim=32)
        q32 = torch.randn(2, 10, 32)
        k32 = torch.randn(2, 10, 32)
        q_out, k_out = rot(q32, k32)
        assert q_out.dtype == torch.float32
        # Same seq_len (10), different dtype -> must recompute, not reuse fp32 cache.
        q_out16, k_out16 = rot(q32.half(), k32.half())
        assert q_out16.dtype == torch.float16
        assert k_out16.dtype == torch.float16
