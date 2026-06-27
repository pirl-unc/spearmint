"""Tests for MintTokenizer and SpearmintTokenizer."""

import torch
import pytest

from mint_stability.tokenizer import MintTokenizer, SpearmintTokenizer


class TestMintTokenizer:
    def setup_method(self):
        self.tok = MintTokenizer()

    def test_prepare_input_shapes(self):
        chains, chain_ids = self.tok.prepare_input("GILGFVFTL", "MAVMAPRTL")
        # peptide: 9 aa + cls + eos = 11, mhc: 9 aa + cls + eos = 11 -> total 22
        assert chains.shape == chain_ids.shape
        assert chains.shape[0] == 22
        assert chains.dtype == torch.int64
        assert chain_ids.dtype == torch.int32

    def test_chain_ids_assignment(self):
        pep = "AAA"
        mhc = "GGG"
        chains, chain_ids = self.tok.prepare_input(pep, mhc)
        # Peptide tokens (5 = cls + 3 aa + eos) should have chain_id 0
        assert (chain_ids[:5] == 0).all()
        # MHC tokens should have chain_id 1
        assert (chain_ids[5:] == 1).all()

    def test_prepare_batch_padding(self):
        peptides = ["AAA", "GGGGGG"]
        mhcs = ["RRR", "DDDDDD"]
        chains, chain_ids = self.tok.prepare_batch(peptides, mhcs)
        assert chains.shape[0] == 2
        assert chains.shape[1] == max(
            5 + 5,  # "AAA" + "RRR" each 3aa + cls + eos
            8 + 8,  # "GGGGGG" + "DDDDDD" each 6aa + cls + eos
        )
        # Shorter sequences should be padded
        assert chains[0, -1] == self.tok.alphabet.padding_idx

    def test_j_replacement(self):
        """J is a non-standard amino acid that should be replaced with L."""
        chains_j, _ = self.tok.prepare_input("JAA", "AAA")
        chains_l, _ = self.tok.prepare_input("LAA", "AAA")
        assert torch.equal(chains_j, chains_l)


class TestSpearmintTokenizer:
    def setup_method(self):
        self.tok = SpearmintTokenizer()

    def test_prepare_input_returns_four_tensors(self):
        chains, chain_ids, assay_idx, temp = self.tok.prepare_input(
            "GILGFVFTL", "MAVMAPRTL", assay="SPA", temperature_c=37.0
        )
        assert chains.shape == chain_ids.shape
        assert assay_idx.shape == (1,)
        assert temp.shape == (1,)

    def test_assay_types(self):
        for assay, expected_idx in [
            ("SPA", 0), ("Purified_Fluor", 1), ("Cellular_Fluor", 2), ("Other", 3),
        ]:
            _, _, assay_idx, _ = self.tok.prepare_input("AAA", "GGG", assay=assay)
            assert assay_idx.item() == expected_idx

    def test_unknown_assay_defaults_to_other(self):
        _, _, assay_idx, _ = self.tok.prepare_input("AAA", "GGG", assay="unknown")
        assert assay_idx.item() == 3  # "Other"

    def test_default_temperature(self):
        _, _, _, temp = self.tok.prepare_input("AAA", "GGG")
        assert temp.item() == 37.0

    def test_prepare_batch_defaults(self):
        peptides = ["AAA", "GGG"]
        mhcs = ["RRR", "DDD"]
        chains, chain_ids, assay_idxs, temps = self.tok.prepare_batch(peptides, mhcs)
        assert chains.shape[0] == 2
        assert assay_idxs.shape == (2,)
        assert temps.shape == (2,)
        # Defaults
        assert (assay_idxs == 0).all()  # SPA
        assert (temps == 37.0).all()

    def test_prepare_batch_custom(self):
        peptides = ["AAA", "GGG"]
        mhcs = ["RRR", "DDD"]
        assays = ["SPA", "Cellular_Fluor"]
        temps_c = [37.0, 25.0]
        _, _, assay_idxs, temps = self.tok.prepare_batch(
            peptides, mhcs, assays=assays, temperatures_c=temps_c
        )
        assert assay_idxs[0].item() == 0
        assert assay_idxs[1].item() == 2
        assert temps[1].item() == 25.0
