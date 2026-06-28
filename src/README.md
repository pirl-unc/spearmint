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
python -m mint_stability.train_binding \
    --data_dir data/binding_affinity \
    --checkpoint_path /path/to/mint.ckpt \
    --use_multimer \
    --task_type reg \
    --num_epochs 20 \
    --lr 5e-4 \
    --bs 32 \
    --freeze_percent 0.5 \
    --device cuda:0 \
    --output_dir checkpoints/stage1_binding \
    --seed 42
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
python -m mint_stability.train_stability \
    --data_dir data/binding_stability \
    --stage1_checkpoint checkpoints/stage1_binding/best_mhc_binding.pt \
    --mint_checkpoint /path/to/mint.ckpt \
    --log_transform \
    --use_multimer \
    --task_type reg \
    --num_epochs 30 \
    --lr 1e-4 \
    --bs 16 \
    --freeze_percent 0.7 \
    --device cuda:0 \
    --output_dir checkpoints/stage2_stability \
    --seed 42
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
# FiLM (Feature-wise Linear Modulation) — recommended
python -m mint_stability.train_stage3 \
    --mode film \
    --data_dir data/stage3_assay_conditioning \
    --stage2_checkpoint checkpoints/stage2_stability/best_stability.pt \
    --output_dir checkpoints/stage3_film \
    --lr 5e-5 \
    --num_epochs 200 \
    --seed 42

# Calibration (per-assay affine + residual MLP)
python -m mint_stability.train_stage3 \
    --mode calibration \
    --data_dir data/stage3_assay_conditioning \
    --stage2_checkpoint checkpoints/stage2_stability/best_stability.pt \
    --output_dir checkpoints/stage3_calibration

# Additive (residual fusion)
python -m mint_stability.train_stage3 \
    --mode additive \
    --data_dir data/stage3_assay_conditioning \
    --stage2_checkpoint checkpoints/stage2_stability/best_stability.pt \
    --output_dir checkpoints/stage3_additive
```

Multi-GPU with Accelerate:
```bash
accelerate launch --multi_gpu --num_processes 4 \
    -m mint_stability.train_stage3 \
    --mode film \
    --data_dir data/stage3_assay_conditioning \
    --stage2_checkpoint checkpoints/stage2_stability/best_stability.pt \
    --output_dir checkpoints/stage3_film
```

Key arguments:
- `--mode`: `film`, `calibration`, or `additive`
- `--stage2_checkpoint`: Path to the Stage 2 `best_stability.pt` checkpoint (a directory is also accepted)
- `--unfreeze_project`: Unfreeze Stage 2 projection head (film/additive only)
- `--ablate_temp`: Zero out temperature embedding (for ablation studies)
- `--loss`: `mse` or `huber` (default: huber)

Output: `checkpoints/stage3_film/best_stage3_film.pt`

## Converting Checkpoints to HuggingFace Format

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

## Running Tests

```bash
# from the repository root
python -m pytest tests/ -v
```

109 tests covering backbone, models, tokenizers, training components, and HF build pipeline.
