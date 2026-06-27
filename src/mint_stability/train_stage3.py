"""
Stage 3: Multi-assay stability prediction with assay/temperature conditioning.

Consolidated training script supporting three conditioning architectures:

  --mode film        FiLM (Feature-wise Linear Modulation): gamma*h + beta
  --mode calibration Per-assay affine calibration + zero-init residual MLP
  --mode additive    Additive residual fusion: h + MLP([h; e_a; e_t])

All modes:
  - Load a trained Stage 2 checkpoint (ESM2 backbone + projection head)
  - Freeze the backbone; optionally freeze the projection head
  - Learn assay/temperature conditioning with identity init (y_hat = y_hat_0 at start)
  - Use log1p(half_life_hours) labels
  - Support single-GPU and multi-GPU via HuggingFace Accelerate

Usage:
    # Single GPU -- FiLM
    python -m mint_stability.train_stage3 --mode film \
        --data_dir data/stage3_v2 \
        --stage2_dir checkpoints/mint_stage2_stability_strict \
        --output_dir checkpoints/stage3_film

    # Multi GPU -- Calibration
    accelerate launch --multi_gpu --num_processes 4 \
        -m mint_stability.train_stage3 --mode calibration \
        --data_dir data/stage3_v2 \
        --stage2_dir checkpoints/mint_stage2_stability_strict \
        --output_dir checkpoints/stage3_calibration

    # Single GPU -- Additive
    python -m mint_stability.train_stage3 --mode additive \
        --data_dir data/stage3_v2 \
        --stage2_dir checkpoints/mint_stage2_stability_strict \
        --output_dir checkpoints/stage3_additive
"""

import argparse
import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    # scikit-learn < 1.4 compat
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)
from torch.utils.data import Dataset
from tqdm import tqdm

from ._compat import torch_load, load_transfer
from .backbone import Alphabet
from .configuration_spearmint import DEFAULT_ASSAY_TYPES, DEFAULT_TEMP_C
from .train_binding import (
    MHCBindingWrapper,
    set_seed,
    load_esm2_config,
    _load_args_from_config,
)

try:
    import wandb
except ImportError:
    wandb = None

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.utils\.data")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*CoW.*|.*copy_on_write.*")


# ---------------------------------------------------------------------------
# Constants (derived from canonical source in configuration_spearmint)
# ---------------------------------------------------------------------------

ASSAY_TYPES = DEFAULT_ASSAY_TYPES
ASSAY_TO_IDX = {a: i for i, a in enumerate(ASSAY_TYPES)}
NUM_ASSAYS = len(ASSAY_TYPES)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Stage3Dataset(Dataset):
    """Dataset for Stage 3 multi-assay stability data.

    Returns 5-tuples: (peptide, mhc, label, assay_idx, temp_float).
    Temperature is passed as a raw float for the scalar linear encoder.
    """

    def __init__(self, df, peptide_col="peptide_sequence", mhc_col="mhc_sequence",
                 target_col="label", assay_col="assay", temp_col="temperature_C"):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.peptide_col = peptide_col
        self.mhc_col = mhc_col
        self.target_col = target_col
        self.assay_col = assay_col
        self.temp_col = temp_col
        self.has_temp = temp_col in df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        assay_idx = ASSAY_TO_IDX.get(row[self.assay_col], ASSAY_TO_IDX["Other"])
        if self.has_temp and not pd.isna(row[self.temp_col]):
            temp_float = float(row[self.temp_col])
        else:
            temp_float = DEFAULT_TEMP_C
        return (row[self.peptide_col], row[self.mhc_col], row[self.target_col],
                assay_idx, temp_float)


class Stage3CollateFn:
    """Collate peptide-MHC pairs with assay indices and temperature floats."""

    def __init__(self, truncation_seq_length=None):
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        peptide_seqs, mhc_seqs, labels, assay_idxs, temp_floats = zip(*batches)
        chains = [self._convert(c) for c in [peptide_seqs, mhc_seqs]]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.tensor(np.array(labels), dtype=torch.float32)
        assay_idxs = torch.tensor(assay_idxs, dtype=torch.long)
        temp_floats = torch.tensor(temp_floats, dtype=torch.float32)
        return chains, chain_ids, labels, assay_idxs, temp_floats

    def _convert(self, seq_str_list):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]
        max_len = max(len(s) for s in seq_encoded_list)
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq
        return tokens


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Stage3FiLM(nn.Module):
    """
    Stage 3 FiLM: feature-conditioned modulation of Stage 2 features.

    Architecture:
        r       = MeanPool(ESM2(x))                          frozen backbone
        h       = ReLU(W1 r + b1)                            trainable or frozen
        u       = [Embed(assay); Linear(temp); h]             feature-conditioned
        [gamma, beta] = MLP_film(u)                           R^hidden_dim each
        h_mod   = gamma * h + beta
        y_hat   = readout(drop(h_mod))                        drop is post-modulation

    Init: gamma=1, beta=0, readout = Stage 2 weights -> y_hat = y_hat_0.

    When unfreeze_project=True, the projection head is trainable, allowing
    the model to learn multi-assay-aware features. Feature-conditioned FiLM
    (h feeds into the MLP alongside metadata) makes gamma,beta sample-specific,
    enabling per-sample modulation rather than per-assay-only modulation.
    """

    def __init__(
        self,
        base_model,
        num_assays=NUM_ASSAYS,
        assay_emb_dim=32,
        temp_emb_dim=16,
        film_hidden_dim=128,
        dropout=0.0,
        unfreeze_project=False,
        ablate_temp=False,
    ):
        super().__init__()
        self.ablate_temp = ablate_temp

        # --- ESM backbone (always frozen) ---
        self.esm = base_model.model
        self.cfg = base_model.cfg
        for p in self.esm.parameters():
            p.requires_grad = False

        # --- Stage 2 projection: Linear + ReLU (dropout moved to after FiLM) ---
        self.project = nn.Sequential(*list(base_model.project.children())[:2])
        hidden_dim = base_model.project[0].out_features
        self._project_frozen = not unfreeze_project
        if self._project_frozen:
            for p in self.project.parameters():
                p.requires_grad = False

        # --- Metadata encoder ---
        self.assay_emb = nn.Embedding(num_assays, assay_emb_dim)
        self.temp_linear = nn.Linear(1, temp_emb_dim)
        if self.ablate_temp:
            for p in self.temp_linear.parameters():
                p.requires_grad = False
        meta_dim = assay_emb_dim + temp_emb_dim

        # --- Feature-conditioned FiLM MLP: [metadata; h] -> [gamma, beta] ---
        film_input_dim = meta_dim + hidden_dim
        self.film_mlp = nn.Sequential(
            nn.Linear(film_input_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, 2 * hidden_dim),
        )

        # --- Post-modulation dropout (replaces project's pre-FiLM dropout) ---
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # --- Readout (init'd from Stage 2) ---
        self.readout = nn.Linear(hidden_dim, 1)
        self.readout.weight.data.copy_(base_model.project[3].weight.data)
        self.readout.bias.data.copy_(base_model.project[3].bias.data)

        # --- FiLM identity init: gamma=1, beta=0 ---
        # Last layer has zero weights -> output is always the bias regardless
        # of input (including h). So at init gamma=1, beta=0 for all samples.
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        with torch.no_grad():
            self.film_mlp[-1].bias[:hidden_dim].fill_(1.0)   # gamma
            # beta already 0

        self._hidden_dim = hidden_dim

    def train(self, mode=True):
        super().train(mode)
        # ESM backbone must always stay in eval mode.
        self.esm.eval()
        # Frozen projection stays in eval mode (no-op for Linear+ReLU,
        # but keeps the contract explicit). When unfrozen, it follows
        # the normal train/eval cycle.
        if self._project_frozen:
            self.project.eval()
        return self

    def encode(self, chains, chain_ids):
        """Frozen ESM2 forward + mean pooling -> [B, 1280]."""
        with torch.no_grad():
            mask = (
                (~chains.eq(self.esm.cls_idx))
                & (~chains.eq(self.esm.eos_idx))
                & (~chains.eq(self.esm.padding_idx))
            )
            chain_out = self.esm(
                chains, chain_ids, repr_layers=[self.cfg.encoder_layers]
            )["representations"][self.cfg.encoder_layers]

            mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
            masked_chain_out = chain_out * mask_expanded
            sum_masked = masked_chain_out.sum(dim=1)
            mask_counts = mask.sum(dim=1, keepdim=True).float()
            return sum_masked / mask_counts

    def forward(self, chains, chain_ids, assay_idxs, temp_floats):
        # ESM backbone (always frozen, no_grad)
        r = self.encode(chains, chain_ids)                      # [B, 1280]

        # Projection: frozen runs under no_grad, trainable runs normally
        if self._project_frozen:
            with torch.no_grad():
                h = self.project(r)                             # [B, hidden_dim]
        else:
            h = self.project(r)                                 # [B, hidden_dim]

        # Feature-conditioned metadata encoding
        e_a = self.assay_emb(assay_idxs)                        # [B, assay_emb_dim]
        e_t = self.temp_linear(temp_floats.unsqueeze(-1))       # [B, temp_emb_dim]
        if self.ablate_temp:
            e_t = torch.zeros_like(e_t)
        u = torch.cat([e_a, e_t, h], dim=-1)                   # [B, meta_dim + hidden_dim]

        # FiLM modulation (gamma,beta are now sample-specific)
        film_out = self.film_mlp(u)                             # [B, 2*hidden_dim]
        gamma = film_out[:, :self._hidden_dim]                  # [B, hidden_dim]
        beta = film_out[:, self._hidden_dim:]                   # [B, hidden_dim]

        h_mod = gamma * h + beta                                # [B, hidden_dim]
        return self.readout(self.drop(h_mod)).squeeze(-1)       # [B]


class Stage3Calibration(nn.Module):
    """
    Stage 3 Calibration: per-assay affine recalibration + zero-init residual.

    Architecture:
        r       = MeanPool(ESM2(x))                       frozen
        h       = ReLU(W1 r + b1)                         frozen (Stage 2 proj, hidden)
        y_0     = w_r^T Dropout(h) + c_r                  frozen (Stage 2 readout)
        e_a     = Embed(assay)                             trainable
        e_t     = Linear(temp)                             trainable
        delta   = MLP_res([h; e_a; e_t])                   trainable, zero-init output
        y_hat   = a_g * y_0 + b_g + delta                  per-assay calibration

    Init: a_g=1, b_g=0, delta=0 -> y_hat = y_0 (exact Stage 2 recovery).
    """

    def __init__(
        self,
        base_model,
        num_assays=NUM_ASSAYS,
        assay_emb_dim=16,
        temp_emb_dim=8,
        residual_hidden_dim=64,
        residual_dropout=0.0,
        ablate_temp=False,
    ):
        super().__init__()
        self.ablate_temp = ablate_temp

        # --- ESM backbone (always frozen) ---
        self.esm = base_model.model
        self.cfg = base_model.cfg
        for p in self.esm.parameters():
            p.requires_grad = False

        # --- Stage 2 full head: frozen (Linear->ReLU->Dropout->Linear) ---
        self.project = base_model.project
        for p in self.project.parameters():
            p.requires_grad = False

        hidden_dim = base_model.project[0].out_features  # 512

        # --- Per-assay affine calibration: a_g (scale) and b_g (bias) ---
        # Init: a=1, b=0 -> y_hat = 1*y_0 + 0 = y_0
        self.calib_scale = nn.Parameter(torch.ones(num_assays))
        self.calib_bias = nn.Parameter(torch.zeros(num_assays))

        # --- Metadata encoder ---
        self.assay_emb = nn.Embedding(num_assays, assay_emb_dim)
        self.temp_linear = nn.Linear(1, temp_emb_dim)
        if self.ablate_temp:
            for p in self.temp_linear.parameters():
                p.requires_grad = False
        meta_dim = assay_emb_dim + temp_emb_dim

        # --- Residual MLP: [h; e_a; e_t] -> scalar correction ---
        # Last layer zero-init so delta=0 at start
        res_input_dim = hidden_dim + meta_dim
        layers = [
            nn.Linear(res_input_dim, residual_hidden_dim),
            nn.ReLU(),
        ]
        if residual_dropout > 0:
            layers.append(nn.Dropout(residual_dropout))
        layers.append(nn.Linear(residual_hidden_dim, 1))
        self.residual_mlp = nn.Sequential(*layers)

        # Zero-init the output layer -> delta=0 at init
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        self._hidden_dim = hidden_dim

    def train(self, mode=True):
        super().train(mode)
        # ESM backbone and Stage 2 head always stay in eval mode
        self.esm.eval()
        self.project.eval()
        return self

    def encode(self, chains, chain_ids):
        """Frozen ESM2 forward + mean pooling -> [B, 1280]."""
        with torch.no_grad():
            mask = (
                (~chains.eq(self.esm.cls_idx))
                & (~chains.eq(self.esm.eos_idx))
                & (~chains.eq(self.esm.padding_idx))
            )
            chain_out = self.esm(
                chains, chain_ids, repr_layers=[self.cfg.encoder_layers]
            )["representations"][self.cfg.encoder_layers]

            mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
            masked_chain_out = chain_out * mask_expanded
            sum_masked = masked_chain_out.sum(dim=1)
            mask_counts = mask.sum(dim=1, keepdim=True).float()
            return sum_masked / mask_counts

    def forward(self, chains, chain_ids, assay_idxs, temp_floats):
        # Frozen ESM backbone
        r = self.encode(chains, chain_ids)                        # [B, 1280]

        # Frozen Stage 2 full head
        with torch.no_grad():
            h = self.project[:2](r)                               # [B, 512] (Linear+ReLU)
            y0 = self.project[2:](h).squeeze(-1)                  # [B] (Dropout+Linear readout)
            h = h.detach()                                        # stop grad explicitly
            y0 = y0.detach()

        # Per-assay calibration: y_calib = a_g * y_0 + b_g
        a = self.calib_scale[assay_idxs]                          # [B]
        b = self.calib_bias[assay_idxs]                           # [B]
        y_calib = a * y0 + b                                      # [B]

        # Residual correction: delta = MLP([h; e_a; e_t])
        e_a = self.assay_emb(assay_idxs)                          # [B, assay_emb_dim]
        e_t = self.temp_linear(temp_floats.unsqueeze(-1))         # [B, temp_emb_dim]
        if self.ablate_temp:
            e_t = torch.zeros_like(e_t)
        u = torch.cat([h, e_a, e_t], dim=-1)                     # [B, hidden+meta]
        delta = self.residual_mlp(u).squeeze(-1)                  # [B]

        return y_calib + delta                                    # [B]


class Stage3Additive(nn.Module):
    """
    Stage 3 Additive: residual fusion with continuous temperature encoding.

    Architecture:
        r       = MeanPool(ESM2(x))                    frozen
        h       = ReLU(W1 r + b1)                      trainable or frozen
        e_a     = Embed(assay)                          trainable
        e_t     = Linear(temp_C)                        trainable (continuous)
        residual = MLP([h; e_a; e_t])                   trainable, zero-init output
        h_fused = h + residual
        y_hat   = readout(drop(h_fused))

    Init: MLP output=0 -> h_fused=h -> y_hat=y_hat_0 (Stage 2 recovery).
    """

    def __init__(
        self,
        base_model,
        num_assays=NUM_ASSAYS,
        assay_emb_dim=32,
        temp_emb_dim=16,
        fusion_dropout=0.0,
        dropout=0.0,
        unfreeze_project=False,
        ablate_temp=False,
    ):
        super().__init__()
        self.ablate_temp = ablate_temp

        # --- ESM backbone (always frozen) ---
        self.esm = base_model.model
        self.cfg = base_model.cfg
        for p in self.esm.parameters():
            p.requires_grad = False

        # --- Stage 2 projection: Linear + ReLU ---
        self.project = nn.Sequential(*list(base_model.project.children())[:2])
        hidden_dim = base_model.project[0].out_features
        self._project_frozen = not unfreeze_project
        if self._project_frozen:
            for p in self.project.parameters():
                p.requires_grad = False

        # --- Metadata encoder ---
        self.assay_emb = nn.Embedding(num_assays, assay_emb_dim)
        self.temp_linear = nn.Linear(1, temp_emb_dim)
        if self.ablate_temp:
            for p in self.temp_linear.parameters():
                p.requires_grad = False
        meta_dim = assay_emb_dim + temp_emb_dim

        # --- Fusion MLP: [h; e_a; e_t] -> hdim residual ---
        # Same shape as old additive fusion but with continuous temp.
        fusion_input_dim = hidden_dim + meta_dim
        fusion_layers = [
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if fusion_dropout > 0:
            fusion_layers.append(nn.Dropout(fusion_dropout))
        fusion_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fusion_mlp = nn.Sequential(*fusion_layers)

        # Zero-init output layer -> residual=0 at init
        nn.init.zeros_(self.fusion_mlp[-1].weight)
        nn.init.zeros_(self.fusion_mlp[-1].bias)

        # --- Post-fusion dropout (before readout) ---
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # --- Readout (init'd from Stage 2) ---
        self.readout = nn.Linear(hidden_dim, 1)
        self.readout.weight.data.copy_(base_model.project[3].weight.data)
        self.readout.bias.data.copy_(base_model.project[3].bias.data)

        self._hidden_dim = hidden_dim

    def train(self, mode=True):
        super().train(mode)
        self.esm.eval()
        if self._project_frozen:
            self.project.eval()
        return self

    def encode(self, chains, chain_ids):
        """Frozen ESM2 forward + mean pooling -> [B, 1280]."""
        with torch.no_grad():
            mask = (
                (~chains.eq(self.esm.cls_idx))
                & (~chains.eq(self.esm.eos_idx))
                & (~chains.eq(self.esm.padding_idx))
            )
            chain_out = self.esm(
                chains, chain_ids, repr_layers=[self.cfg.encoder_layers]
            )["representations"][self.cfg.encoder_layers]

            mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
            masked_chain_out = chain_out * mask_expanded
            sum_masked = masked_chain_out.sum(dim=1)
            mask_counts = mask.sum(dim=1, keepdim=True).float()
            return sum_masked / mask_counts

    def forward(self, chains, chain_ids, assay_idxs, temp_floats):
        # ESM backbone (always frozen, no_grad)
        r = self.encode(chains, chain_ids)                      # [B, 1280]

        # Projection: frozen runs under no_grad, trainable runs normally
        if self._project_frozen:
            with torch.no_grad():
                h = self.project(r)                             # [B, hidden_dim]
        else:
            h = self.project(r)                                 # [B, hidden_dim]

        # Metadata encoding (continuous temperature)
        e_a = self.assay_emb(assay_idxs)                        # [B, assay_emb_dim]
        e_t = self.temp_linear(temp_floats.unsqueeze(-1))       # [B, temp_emb_dim]
        if self.ablate_temp:
            e_t = torch.zeros_like(e_t)

        # Additive residual fusion
        u = torch.cat([h, e_a, e_t], dim=-1)                   # [B, hidden+meta]
        residual = self.fusion_mlp(u)                           # [B, hidden_dim]
        h_fused = h + residual                                  # [B, hidden_dim]

        return self.readout(self.drop(h_fused)).squeeze(-1)     # [B]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(targets, predictions, name):
    rmse = root_mean_squared_error(targets, predictions)
    pearson_r, _ = pearsonr(targets, predictions)
    spearman_r, _ = spearmanr(targets, predictions)
    return {
        f"{name}_RMSE": rmse,
        f"{name}_Pearson": pearson_r,
        f"{name}_Spearman": spearman_r,
    }


def per_assay_metrics(targets, predictions, assay_indices, name):
    metrics = {}
    for assay_name, assay_idx in ASSAY_TO_IDX.items():
        mask = assay_indices == assay_idx
        if mask.sum() < 5:
            continue
        t, p = targets[mask], predictions[mask]
        metrics[f"{name}_{assay_name}_RMSE"] = root_mean_squared_error(t, p)
        if mask.sum() >= 10:
            sr, _ = spearmanr(t, p)
            pr, _ = pearsonr(t, p)
            metrics[f"{name}_{assay_name}_Spearman"] = sr
            metrics[f"{name}_{assay_name}_Pearson"] = pr
    return metrics


def make_loss_fn(args):
    if args.loss == "huber":
        return nn.HuberLoss(delta=args.huber_delta)
    return nn.MSELoss()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, args, prefix, accelerator):
    model.eval()
    loss_fn = make_loss_fn(args)

    preds_list, targets_list, assay_list = [], [], []
    loss_accum = 0
    n_batches = 0
    disable_tqdm = not accelerator.is_main_process

    for batch in tqdm(loader, desc=f"Eval {prefix}", leave=False, disable=disable_tqdm):
        chains, chain_ids, target, assay_idxs, temp_floats = batch
        logits = model(chains, chain_ids, assay_idxs, temp_floats)
        loss_accum += loss_fn(logits, target).item()
        n_batches += 1

        gathered_preds = accelerator.gather_for_metrics(logits)
        gathered_targets = accelerator.gather_for_metrics(target)
        gathered_assays = accelerator.gather_for_metrics(assay_idxs)
        preds_list.append(gathered_preds.cpu().numpy())
        targets_list.append(gathered_targets.cpu().numpy())
        assay_list.append(gathered_assays.cpu().numpy())

    preds = np.concatenate(preds_list).ravel()
    targets = np.concatenate(targets_list).ravel()
    assays = np.concatenate(assay_list).ravel()

    # Synchronize loss across ranks
    loss_tensor = torch.tensor([loss_accum / max(n_batches, 1)],
                               device=accelerator.device)
    loss_tensor = accelerator.reduce(loss_tensor, reduction="mean")
    avg_loss = loss_tensor.item()

    metrics = regression_metrics(targets, preds, prefix)
    metrics.update(per_assay_metrics(targets, preds, assays, prefix))
    metrics[f"{prefix}_loss"] = avg_loss
    return metrics


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, test_loader, cfg, args,
          accelerator, use_wandb=False):
    accum = args.grad_accum_steps

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=json.loads(cfg.adam_betas),
        eps=cfg.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    loss_fn = make_loss_fn(args)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    best_metric = float("inf")
    best_metrics = {}
    patience_counter = 0
    global_step = 0
    micro_step = 0
    running_loss = 0
    is_main = accelerator.is_main_process

    best_ckpt_name = f"best_stage3_{args.mode}.pt"
    final_ckpt_name = f"final_stage3_{args.mode}.pt"

    def _run_eval(epoch, global_step, train_loss_avg):
        nonlocal best_metric, best_metrics, patience_counter

        val_metrics = evaluate(model, val_loader, args, "val", accelerator)
        test_metrics = evaluate(model, test_loader, args, "test", accelerator)
        metrics = {**val_metrics, **test_metrics, "train_loss": train_loss_avg}

        if is_main:
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v:.4f}")
            if use_wandb:
                wandb.log({"epoch": epoch, **metrics}, step=global_step)

        current = metrics["val_loss"]
        if current < best_metric:
            best_metric = current
            best_metrics = metrics
            patience_counter = 0
            if is_main:
                save_path = os.path.join(args.output_dir, best_ckpt_name)
                torch.save({
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "epoch": epoch,
                    "best_metrics": best_metrics,
                    "args": vars(args),
                    "mode": args.mode,
                }, save_path)
                print(f"  -> Saved best model (val_loss={current:.4f})")
        else:
            patience_counter += 1

        model.train()
        should_stop = args.early_stopping > 0 and patience_counter >= args.early_stopping
        stop_tensor = torch.tensor([int(should_stop)], device=accelerator.device)
        stop_tensor = accelerator.reduce(stop_tensor, reduction="sum")
        return stop_tensor.item() > 0

    disable_tqdm = not is_main

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss_accum = 0
        epoch_batches = 0
        done = False

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", disable=disable_tqdm):
            chains, chain_ids, target, assay_idxs, temp_floats = batch
            pred = model(chains, chain_ids, assay_idxs, temp_floats)
            loss = loss_fn(pred, target) / accum
            accelerator.backward(loss)

            raw_loss = loss.item() * accum
            running_loss += raw_loss
            epoch_loss_accum += raw_loss
            epoch_batches += 1
            micro_step += 1

            if micro_step % accum == 0:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_accum_loss = running_loss / accum
                running_loss = 0

                if global_step % args.log_steps == 0 and is_main:
                    print(f"  step {global_step} | loss: {avg_accum_loss:.4f}")
                    if use_wandb:
                        wandb.log({"train_loss_step": avg_accum_loss}, step=global_step)

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    avg_so_far = epoch_loss_accum / epoch_batches
                    if is_main:
                        print(f"Eval at step {global_step} (epoch {epoch})")
                    stopped = _run_eval(epoch, global_step, avg_so_far)
                    if stopped:
                        if is_main:
                            print(f"Early stopping at step {global_step}")
                        done = True
                        break

                if args.max_steps > 0 and global_step >= args.max_steps:
                    if is_main:
                        print(f"Reached max_steps={args.max_steps}")
                    done = True
                    break

        if micro_step % accum != 0 and not done:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            running_loss = 0

        avg_loss = epoch_loss_accum / max(epoch_batches, 1)
        scheduler.step(avg_loss)
        if is_main:
            print(f"Epoch {epoch} | Train loss: {avg_loss:.4f} | Steps: {global_step}")

        if done:
            break

        if is_main:
            print(f"End-of-epoch {epoch} evaluation:")
        stopped = _run_eval(epoch, global_step, avg_loss)
        if stopped:
            if is_main:
                print(f"Early stopping at epoch {epoch}")
            break

    # Save final checkpoint
    if is_main:
        final_path = os.path.join(args.output_dir, final_ckpt_name)
        torch.save({
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "global_step": global_step,
            "metrics": best_metrics,
            "args": vars(args),
            "mode": args.mode,
        }, final_path)

        print(f"\n--- Best Stage 3 ({args.mode}) metrics ---")
        for k, v in sorted(best_metrics.items()):
            print(f"  {k}: {v:.4f}")

    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if Accelerator is None:
        raise ImportError(
            "accelerate is required for Stage 3 training. Install with: pip install accelerate"
        )

    parser = argparse.ArgumentParser(
        description="Stage 3: multi-assay stability prediction"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (values in 'args' key set defaults)")

    # Mode
    parser.add_argument("--mode", type=str, default=None,
                        choices=["film", "calibration", "additive"],
                        help="Conditioning architecture")

    # Data
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--peptide_col", type=str, default="peptide_sequence")
    parser.add_argument("--mhc_col", type=str, default="mhc_sequence")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--assay_col", type=str, default="assay")
    parser.add_argument("--temp_col", type=str, default="temperature_C")

    # Checkpoints
    parser.add_argument("--stage2_dir", type=str, default=None)
    parser.add_argument("--mint_checkpoint", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)

    # Architecture (mode-specific)
    parser.add_argument("--assay_emb_dim", type=int, default=32)
    parser.add_argument("--temp_emb_dim", type=int, default=16)
    parser.add_argument("--film_hidden_dim", type=int, default=128,
                        help="FiLM MLP hidden dim (film mode only)")
    parser.add_argument("--unfreeze_project", action="store_true",
                        help="Unfreeze Stage 2 projection head (film/additive)")
    parser.add_argument("--residual_hidden_dim", type=int, default=64,
                        help="Residual MLP hidden dim (calibration mode only)")
    parser.add_argument("--residual_dropout", type=float, default=0.0,
                        help="Dropout inside residual MLP (calibration mode only)")
    parser.add_argument("--fusion_dropout", type=float, default=0.0,
                        help="Dropout inside fusion MLP (additive mode only)")
    parser.add_argument("--ablate_temp", action="store_true",
                        help="Zero out temperature embedding (ablation study)")

    # Training
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Post-modulation dropout before readout (film/additive)")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--truncation_len", type=int, default=None)
    parser.add_argument("--val_subsample", type=int, default=0,
                        help="Subsample val/test to this many rows (0=use all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable cudnn deterministic mode (slower but bit-reproducible on GPU)")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage3")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = _load_args_from_config(parser)
    if args.mode is None:
        parser.error("--mode is required (via CLI or --config)")
    if args.data_dir is None:
        parser.error("--data_dir is required (via CLI or --config)")
    if args.stage2_dir is None:
        parser.error("--stage2_dir is required (via CLI or --config)")

    # Warn on mode-specific args that don't apply to the chosen mode
    _MODE_SPECIFIC = {
        "film": {"film_hidden_dim", "unfreeze_project", "dropout"},
        "calibration": {"residual_hidden_dim", "residual_dropout"},
        "additive": {"fusion_dropout", "unfreeze_project", "dropout"},
    }
    _ALL_MODE_ARGS = set().union(*_MODE_SPECIFIC.values())
    _applicable = _MODE_SPECIFIC.get(args.mode, set())
    for arg_name in _ALL_MODE_ARGS - _applicable:
        default = parser.get_default(arg_name)
        current = getattr(args, arg_name, default)
        if current != default:
            import warnings
            warnings.warn(
                f"--{arg_name}={current} has no effect in --mode={args.mode}",
                stacklevel=1,
            )

    accelerator = Accelerator()
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    # Seed
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))

    # -----------------------------------------------------------------------
    # Load Stage 2 checkpoint
    # -----------------------------------------------------------------------
    stage2_path = os.path.join(args.stage2_dir, "best_stability.pt")
    stage2_ckpt = torch_load(stage2_path, map_location="cpu", weights_only=False)
    s2_args = stage2_ckpt.get("args", {})

    use_multimer = s2_args.get("use_multimer", True)
    hdim = s2_args.get("hdim", 512)
    s2_dropout = s2_args.get("dropout", 0.2)
    freeze_percent = s2_args.get("freeze_percent", 0.7)
    s2_sigmoid = s2_args.get("stability_score", False)

    if s2_sigmoid and is_main:
        print(
            "WARNING: Stage 2 was trained with --stability_score (sigmoid output). "
            "Stage 3 uses log1p(hours) labels without sigmoid. The S2 readout "
            "weights may be in the wrong scale — consider retraining S2 without "
            "--stability_score, or verifying that identity init compensates."
        )

    if args.mint_checkpoint is None:
        args.mint_checkpoint = s2_args.get("mint_checkpoint")
        if args.mint_checkpoint is None:
            for candidate in ["../../mint.ckpt", "../../../mint.ckpt"]:
                p = os.path.join(os.path.dirname(os.path.abspath(__file__)), candidate)
                if os.path.exists(p):
                    args.mint_checkpoint = p
                    break
        if args.mint_checkpoint is None:
            raise ValueError("Cannot find MINT checkpoint.")

    if is_main:
        print(f"Stage 2 config: use_multimer={use_multimer}, hdim={hdim}, "
              f"dropout={s2_dropout}, freeze_percent={freeze_percent}")

    # Load ESM2 config
    cfg = load_esm2_config(args.config_path)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    has_temp = args.temp_col in train_df.columns

    if args.val_subsample > 0:
        if args.val_subsample < len(val_df):
            val_df = val_df.sample(n=args.val_subsample, random_state=42)
        if args.val_subsample < len(test_df):
            test_df = test_df.sample(n=args.val_subsample, random_state=42)

    if is_main:
        if not has_temp:
            print(f"Warning: '{args.temp_col}' not found — using {DEFAULT_TEMP_C}C.")
        print(f"Stage 3 ({args.mode}) data: train={len(train_df)}, "
              f"val={len(val_df)}, test={len(test_df)}")
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            print(f"  {split_name} assay: {df[args.assay_col].value_counts().to_dict()}")

    # Target transform: log1p(hours)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    for df in [train_df, val_df, test_df]:
        df[args.label_col] = np.log1p(df[args.label_col].values.clip(min=0))
    if is_main:
        print("Applied log1p(hours) transform to labels")

    # -----------------------------------------------------------------------
    # Datasets and loaders
    # -----------------------------------------------------------------------
    collate_fn = Stage3CollateFn(truncation_seq_length=args.truncation_len)

    train_dataset = Stage3Dataset(
        train_df, args.peptide_col, args.mhc_col, args.label_col,
        args.assay_col, args.temp_col,
    )
    val_dataset = Stage3Dataset(
        val_df, args.peptide_col, args.mhc_col, args.label_col,
        args.assay_col, args.temp_col,
    )
    test_dataset = Stage3Dataset(
        test_df, args.peptide_col, args.mhc_col, args.label_col,
        args.assay_col, args.temp_col,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True,
    )
    eval_bs = args.eval_bs if args.eval_bs > 0 else args.bs
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eval_bs, collate_fn=collate_fn, shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_bs, collate_fn=collate_fn, shuffle=False,
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    base_model = MHCBindingWrapper(
        cfg,
        args.mint_checkpoint,
        freeze_percent=freeze_percent,
        use_multimer=use_multimer,
        hidden_dim=hdim,
        dropout=s2_dropout,
        output_size=1,
        device="cpu",
        sigmoid_output=False,
    )

    s2_state = stage2_ckpt["model_state_dict"]
    missing = load_transfer(base_model, s2_state, label="S2→S3 base")
    if is_main:
        if missing:
            print(f"Warning: missing keys when loading Stage 2: {missing}")
        print("Loaded Stage 2 checkpoint into base model")

    if args.mode == "film":
        model = Stage3FiLM(
            base_model,
            num_assays=NUM_ASSAYS,
            assay_emb_dim=args.assay_emb_dim,
            temp_emb_dim=args.temp_emb_dim,
            film_hidden_dim=args.film_hidden_dim,
            dropout=args.dropout,
            unfreeze_project=args.unfreeze_project,
            ablate_temp=args.ablate_temp,
        )
    elif args.mode == "calibration":
        model = Stage3Calibration(
            base_model,
            num_assays=NUM_ASSAYS,
            assay_emb_dim=args.assay_emb_dim,
            temp_emb_dim=args.temp_emb_dim,
            residual_hidden_dim=args.residual_hidden_dim,
            residual_dropout=args.residual_dropout,
            ablate_temp=args.ablate_temp,
        )
    elif args.mode == "additive":
        model = Stage3Additive(
            base_model,
            num_assays=NUM_ASSAYS,
            assay_emb_dim=args.assay_emb_dim,
            temp_emb_dim=args.temp_emb_dim,
            fusion_dropout=args.fusion_dropout,
            dropout=args.dropout,
            unfreeze_project=args.unfreeze_project,
            ablate_temp=args.ablate_temp,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Mode: {args.mode} | Parameters: {trainable:,} trainable / "
              f"{total:,} total ({100*trainable/total:.1f}%)")
        if args.ablate_temp:
            print("*** ABLATION: temperature embedding zeroed out (--ablate_temp) ***")

    # -----------------------------------------------------------------------
    # Weights & Biases (main process only)
    # -----------------------------------------------------------------------
    use_wandb = args.wandb_project is not None and is_main
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when --wandb_project is set. Install with: pip install wandb")
        run_name = args.wandb_run_name or f"stage3_{args.mode}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={**vars(args), "stage2_args": s2_args},
        )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    try:
        train(model, train_loader, val_loader, test_loader, cfg, args,
              accelerator, use_wandb=use_wandb)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if is_main:
                print("\n*** OOM: this config exceeds GPU memory. ***")
            torch.cuda.empty_cache()
            if use_wandb:
                wandb.log({"val_loss": 1e6, "oom": 1})
        else:
            raise

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
