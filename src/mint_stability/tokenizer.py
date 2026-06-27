"""Tokenizers for MINT stability models."""

from typing import List, Optional, Tuple

import torch

from .backbone import Alphabet
from .configuration_spearmint import (
    DEFAULT_ASSAY_TYPES,
    DEFAULT_TEMP_C as _DEFAULT_TEMP_C,
)

__all__ = ["MintTokenizer", "SpearmintTokenizer"]


class MintTokenizer:
    """Tokenize peptide + MHC sequences for MINT S1/S2 models.

    Usage:
        tokenizer = MintTokenizer()
        chains, chain_ids = tokenizer.prepare_input("GILGFVFTL", "MAVMAPRTL...")
        output = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
    """

    def __init__(self):
        self.alphabet = Alphabet.from_architecture("ESM-1b")

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        seq = seq.replace("J", "L")
        encoded = self.alphabet.encode("<cls>" + seq + "<eos>")
        return torch.tensor(encoded, dtype=torch.int64)

    def prepare_input(
        self, peptide: str, mhc_sequence: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a single peptide-MHC pair.

        Returns:
            chains: Token IDs, shape (seq_len,). Add batch dim with .unsqueeze(0).
            chain_ids: Chain membership, shape (seq_len,). 0=peptide, 1=MHC.
        """
        pep_tokens = self._encode_sequence(peptide)
        mhc_tokens = self._encode_sequence(mhc_sequence)

        chains = torch.cat([pep_tokens, mhc_tokens], dim=0)
        chain_ids = torch.cat(
            [
                torch.zeros(len(pep_tokens), dtype=torch.int32),
                torch.ones(len(mhc_tokens), dtype=torch.int32),
            ],
            dim=0,
        )
        return chains, chain_ids

    def prepare_batch(
        self, peptides: List[str], mhc_sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of peptide-MHC pairs with padding.

        Returns:
            chains: (batch, max_seq_len)
            chain_ids: (batch, max_seq_len)
        """
        all_chains = []
        all_chain_ids = []
        for pep, mhc in zip(peptides, mhc_sequences):
            c, cid = self.prepare_input(pep, mhc)
            all_chains.append(c)
            all_chain_ids.append(cid)

        max_len = max(len(c) for c in all_chains)
        padded_chains = torch.full(
            (len(all_chains), max_len),
            self.alphabet.padding_idx,
            dtype=torch.int64,
        )
        padded_chain_ids = torch.zeros(
            (len(all_chains), max_len), dtype=torch.int32
        )

        for i, (c, cid) in enumerate(zip(all_chains, all_chain_ids)):
            padded_chains[i, : len(c)] = c
            padded_chain_ids[i, : len(cid)] = cid

        return padded_chains, padded_chain_ids


class SpearmintTokenizer:
    """Tokenize peptide + MHC sequences for SPEARMINT S3 models.

    Usage:
        tokenizer = SpearmintTokenizer()
        chains, chain_ids, assay_idx, temp = tokenizer.prepare_input(
            "GILGFVFTL", "MAVMAPRTL...", assay="SPA", temperature_c=37.0
        )
        output = model(
            chains.unsqueeze(0), chain_ids.unsqueeze(0),
            assay_idx, temp,
        )
    """

    ASSAY_TYPES = DEFAULT_ASSAY_TYPES
    ASSAY_TO_IDX = {a: i for i, a in enumerate(DEFAULT_ASSAY_TYPES)}
    DEFAULT_TEMP_C = _DEFAULT_TEMP_C

    def __init__(self):
        self.alphabet = Alphabet.from_architecture("ESM-1b")

    def _encode_sequence(self, seq: str) -> torch.Tensor:
        seq = seq.replace("J", "L")
        encoded = self.alphabet.encode("<cls>" + seq + "<eos>")
        return torch.tensor(encoded, dtype=torch.int64)

    def prepare_input(
        self,
        peptide: str,
        mhc_sequence: str,
        assay: str = "SPA",
        temperature_c: float = 37.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a single peptide-MHC pair with metadata.

        Returns:
            chains: Token IDs, shape (seq_len,).
            chain_ids: Chain membership, shape (seq_len,). 0=peptide, 1=MHC.
            assay_idx: Assay type index, shape (1,).
            temp_float: Temperature in Celsius, shape (1,).
        """
        pep_tokens = self._encode_sequence(peptide)
        mhc_tokens = self._encode_sequence(mhc_sequence)

        chains = torch.cat([pep_tokens, mhc_tokens], dim=0)
        chain_ids = torch.cat(
            [
                torch.zeros(len(pep_tokens), dtype=torch.int32),
                torch.ones(len(mhc_tokens), dtype=torch.int32),
            ],
            dim=0,
        )
        assay_idx = torch.tensor(
            [self.ASSAY_TO_IDX.get(assay, self.ASSAY_TO_IDX["Other"])],
            dtype=torch.long,
        )
        temp_float = torch.tensor([temperature_c], dtype=torch.float32)
        return chains, chain_ids, assay_idx, temp_float

    def prepare_batch(
        self,
        peptides: List[str],
        mhc_sequences: List[str],
        assays: Optional[List[str]] = None,
        temperatures_c: Optional[List[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a batch of peptide-MHC pairs with padding.

        Returns:
            chains: (batch, max_seq_len)
            chain_ids: (batch, max_seq_len)
            assay_idxs: (batch,)
            temp_floats: (batch,)
        """
        if assays is None:
            assays = ["SPA"] * len(peptides)
        if temperatures_c is None:
            temperatures_c = [self.DEFAULT_TEMP_C] * len(peptides)

        all_chains = []
        all_chain_ids = []
        for pep, mhc in zip(peptides, mhc_sequences):
            pep_tokens = self._encode_sequence(pep)
            mhc_tokens = self._encode_sequence(mhc)
            chains = torch.cat([pep_tokens, mhc_tokens], dim=0)
            chain_ids = torch.cat(
                [
                    torch.zeros(len(pep_tokens), dtype=torch.int32),
                    torch.ones(len(mhc_tokens), dtype=torch.int32),
                ],
                dim=0,
            )
            all_chains.append(chains)
            all_chain_ids.append(chain_ids)

        max_len = max(len(c) for c in all_chains)
        padded_chains = torch.full(
            (len(all_chains), max_len),
            self.alphabet.padding_idx,
            dtype=torch.int64,
        )
        padded_chain_ids = torch.zeros(
            (len(all_chains), max_len), dtype=torch.int32
        )

        for i, (c, cid) in enumerate(zip(all_chains, all_chain_ids)):
            padded_chains[i, : len(c)] = c
            padded_chain_ids[i, : len(cid)] = cid

        assay_idxs = torch.tensor(
            [self.ASSAY_TO_IDX.get(a, self.ASSAY_TO_IDX["Other"]) for a in assays],
            dtype=torch.long,
        )
        temp_floats = torch.tensor(temperatures_c, dtype=torch.float32)

        return padded_chains, padded_chain_ids, assay_idxs, temp_floats
