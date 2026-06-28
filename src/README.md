# MINT Stability Training

Training scripts for the MINT peptide-MHC stability prediction pipeline. All three stages share the ESM2-650M backbone from `mint_stability.backbone`.

## Prerequisites

```bash
# From the repository root
pip install -e src/

# Optional (only needed at runtime, not for imports/tests)
pip install wandb           # for experiment tracking (--wandb_project)
pip install accelerate      # for multi-GPU Stage 3 training
```

## Running Tests

```bash
# from the repository root
python -m pytest tests/ -v
```

~100 tests covering backbone, models, tokenizers, training components, and HF build pipeline.

## Data Format

All stages expect `train.csv`, `val.csv`, `test.csv` in a data directory.

**Stage 1 (Binding)** — columns: `peptide_sequence`, `mhc_sequence`, `label`
```
peptide_sequence,mhc_sequence,label
GILGFVFTL,MAVMAPRTLVLLLSGALALTQTWAG...,1
```

**Stage 2 (Stability)** — same columns, `label` is stability score (e.g. half-life hours)
```
peptide_sequence,mhc_sequence,label
GILGFVFTL,MAVMAPRTLVLLLSGALALTQTWAG...,1.8
```

**Stage 3 (Multi-assay)** — adds `assay` and `temperature_C` columns
```
peptide_sequence,mhc_sequence,label,assay,temperature_C
RMPEAAPPV,MAVMAPRTLVLLLSGALALTQTWAG...,6.0,Cellular_Fluor,37.0
```

Assay types: `SPA`, `Purified_Fluor`, `Cellular_Fluor`, `Other`. Temperature defaults to 37.0 if missing.

The `mhc_sequence` column should contain the **full MHC alpha chain** (~365 residues for class I), not a pseudo-sequence. MINT's multimer attention operates on residue-level cross-chain interactions.

## Stage 1: Binding Affinity

Fine-tune MINT backbone on peptide-MHC binding affinity data.

```bash
# Reproduces the published Stage 1 run — all hyperparameters live in the config.
# Supply your local MINT base checkpoint via --checkpoint_path (CLI flags override the config):
python -m mint_stability.train_binding --config configs/s1_binding_args.json \
    --checkpoint_path /path/to/mint.ckpt
```

Key arguments:
- `--checkpoint_path`: Path to the pretrained MINT checkpoint (mint.ckpt)
- `--task_type`: `cls` (binary classification with BCE loss) or `reg` (regression with MSE loss)
- `--use_multimer`: Enable cross-chain multimer attention (recommended)
- `--freeze_percent`: Fraction of backbone layers to freeze (0.0=none, 1.0=all)
- `--config_path`: Optional path to custom ESM2 config JSON (defaults to bundled config)

Output: `checkpoints/stage1_binding/best_mhc_binding.pt`

## Stage 2: Stability Prediction

Double fine-tune on stability data, loading the Stage 1 checkpoint.

```bash
# Reproduces the published Stage 2 run (transfers from Stage 1; log1p labels etc. all in the config):
python -m mint_stability.train_stability --config configs/s2_stability_args.json \
    --mint_checkpoint /path/to/mint.ckpt
```

Key arguments:
- `--stage1_checkpoint`: Path to Stage 1 best checkpoint
- `--mint_checkpoint`: Path to original MINT checkpoint (for architecture init)
- `--reinit_head`: Reinitialize the projection head (recommended if switching task type)
- `--log_transform`: Apply log(1+x) to labels before training
- `--stability_score`: Transform labels to stability score s = 2^(-1/T_half) in [0,1]

Output: `checkpoints/stage2_stability/best_stability.pt`

## Stage 3: Multi-Assay (SPEARMINT)

Train assay/temperature-conditioned models on top of Stage 2. Three conditioning modes:

```bash
# FiLM — the released SPEARMINT model (emb dims, unfreeze_project, loss, etc. all in the config)
python -m mint_stability.train_stage3 --config configs/s3_film_v2_args.json \
    --mint_checkpoint /path/to/mint.ckpt

# Calibration variant
python -m mint_stability.train_stage3 --config configs/s3_calibration_v2_args.json \
    --mint_checkpoint /path/to/mint.ckpt

# (Additive is also a supported `--mode additive`, without a bundled v2 config.)
```

Multi-GPU with Accelerate:
```bash
accelerate launch --multi_gpu --num_processes 4 \
    -m mint_stability.train_stage3 --config configs/s3_film_v2_args.json \
    --mint_checkpoint /path/to/mint.ckpt
```

Key arguments:
- `--mode`: `film`, `calibration`, or `additive`
- `--stage2_checkpoint`: Path to the Stage 2 `best_stability.pt` checkpoint (a directory is also accepted)
- `--unfreeze_project`: Unfreeze Stage 2 projection head (film/additive only)
- `--ablate_temp`: Zero out temperature embedding (for ablation studies)
- `--loss`: `mse` or `huber` (default: huber)

Output: `checkpoints/stage3_film/best_stage3_film.pt`

## Deployment

After training, convert checkpoints to HF format for distribution:

```bash
# Stage 2
python -m mint_stability.convert_checkpoint \
    --stage s2 \
    --checkpoint_path checkpoints/stage2_stability/best_stability.pt \
    --output_dir ./mint-2stage-stability \
    --verify

# Stage 3
python -m mint_stability.convert_checkpoint \
    --stage s3 \
    --checkpoint_path checkpoints/stage3_film/best_stage3_film.pt \
    --output_dir ./spearmint \
    --verify
```
