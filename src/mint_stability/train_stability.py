"""
Stage 2: Double fine-tune on peptide-MHC stability prediction.

Loads the Stage 1 checkpoint (binding affinity model) and continues training
on the smaller stability dataset. The backbone retains learned binding
representations while the head is re-initialized for stability prediction.

Expected CSV format (train/val/test):
    peptide_sequence  - Peptide amino acid sequence (str)
    mhc_sequence      - FULL MHC alpha chain sequence (str, ~365 residues for class I)
                         Use the complete protein sequence, NOT a pseudo-sequence.
                         MINT's multimer attention operates on residue-level cross-chain
                         interactions - the full sequence is needed to capture these.
    label             - Stability label:
                          For classification: 0/1 (unstable/stable)
                          For regression: continuous stability score (e.g. half-life hours)

Usage:
    python -m mint_stability.train_stability \
        --data_dir /path/to/stability_data \
        --stage1_checkpoint checkpoints/stage1_mhc_binding/best_mhc_binding.pt \
        --mint_checkpoint /path/to/mint.ckpt \
        --use_multimer \
        --task_type reg \
        --num_epochs 30 \
        --lr 1e-4 \
        --bs 16 \
        --device cuda:0
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ._compat import torch_load, load_transfer
from .train_binding import (
    MHCBindingDataset,
    MHCCollateFn,
    MHCBindingWrapper,
    classification_metrics,
    regression_metrics,
    set_seed,
    load_esm2_config,
    _load_args_from_config,
)

try:
    import wandb
except ImportError:
    wandb = None

warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.utils\.data")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*CoW.*|.*copy_on_write.*")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unpack_batch(batch, device):
    """Unpack batch to (chains, chain_ids, target) on device."""
    chains, chain_ids, target = batch
    return chains.to(device), chain_ids.to(device), target.to(device)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, args, prefix):
    model.eval()
    device = args.device
    if args.task_type == "cls":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    elif getattr(args, "loss", "mse") == "huber":
        loss_fn = torch.nn.HuberLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    preds, targets = [], []
    loss_accum = 0
    n_batches = 0
    for batch in tqdm(loader, desc=f"Eval {prefix}"):
        chains, chain_ids, target = _unpack_batch(batch, device)
        logits = model(chains, chain_ids)
        loss_accum += loss_fn(logits.squeeze(-1), target).item()
        n_batches += 1
        if args.task_type == "cls":
            pred = torch.sigmoid(logits)
        else:
            pred = logits
        preds.append(pred.squeeze(-1).detach().cpu().numpy())
        targets.append(target.cpu().numpy())

    preds = np.concatenate(preds).ravel()
    targets = np.concatenate(targets).ravel()
    avg_loss = loss_accum / max(n_batches, 1)

    if args.task_type == "cls":
        metrics = classification_metrics(targets, preds, prefix)
    else:
        metrics = regression_metrics(targets, preds, prefix)
    metrics[f"{prefix}_loss"] = avg_loss
    return metrics


# ---------------------------------------------------------------------------
# Train Stage 2
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, test_loader, cfg, args, use_wandb=False):
    device = args.device
    accum = args.grad_accum_steps

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=json.loads(cfg.adam_betas),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )

    if args.task_type == "cls":
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif getattr(args, "loss", "mse") == "huber":
        loss_fn = torch.nn.HuberLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    monitor_metric = "val_loss"
    monitor_higher_better = False

    model.to(device)

    best_metric = float("inf")
    best_metrics = {}
    patience_counter = 0
    global_step = 0
    micro_step = 0
    running_loss = 0

    def _run_eval(epoch, global_step, train_loss_avg):
        """Run evaluation and return whether we should early-stop."""
        nonlocal best_metric, best_metrics, patience_counter

        val_metrics = evaluate(model, val_loader, args, "val")
        test_metrics = evaluate(model, test_loader, args, "test")
        metrics = {**val_metrics, **test_metrics, "train_loss": train_loss_avg}

        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        if use_wandb:
            wandb.log({"epoch": epoch, **metrics}, step=global_step)

        current = metrics[monitor_metric]
        improved = (current > best_metric) if monitor_higher_better else (current < best_metric)
        if improved:
            best_metric = current
            best_metrics = metrics
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "best_stability.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_metrics": best_metrics,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"  -> Saved best model ({monitor_metric}={current:.4f})")
        else:
            patience_counter += 1

        model.train()
        return args.early_stopping > 0 and patience_counter >= args.early_stopping

    done = False
    epoch_loss_accum = 0
    epoch_batches = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss_accum = 0
        epoch_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            chains, chain_ids, target = _unpack_batch(batch, device)
            pred = model(chains, chain_ids)
            loss = loss_fn(pred.squeeze(-1), target) / accum
            loss.backward()

            raw_loss = loss.item() * accum
            running_loss += raw_loss
            epoch_loss_accum += raw_loss
            epoch_batches += 1
            micro_step += 1

            if micro_step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                avg_accum_loss = running_loss / accum
                running_loss = 0

                if global_step % args.log_steps == 0:
                    print(f"  step {global_step} | loss: {avg_accum_loss:.4f}")
                    if use_wandb:
                        wandb.log({"train_loss_step": avg_accum_loss}, step=global_step)

                if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                    avg_so_far = epoch_loss_accum / epoch_batches
                    print(f"Eval at step {global_step} (epoch {epoch})")
                    stopped = _run_eval(epoch, global_step, avg_so_far)
                    if stopped:
                        print(f"Early stopping at step {global_step}")
                        done = True
                        break

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"Reached max_steps={args.max_steps}")
                    done = True
                    break

        if micro_step % accum != 0 and not done:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            running_loss = 0

        avg_loss = epoch_loss_accum / max(epoch_batches, 1)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} | Train loss: {avg_loss:.4f} | Steps: {global_step}")

        if done:
            break

        print(f"End-of-epoch {epoch} evaluation:")
        stopped = _run_eval(epoch, global_step, avg_loss)
        if stopped:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "final_stability.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "global_step": global_step,
            "metrics": best_metrics,
            "args": vars(args),
        },
        final_path,
    )

    print("\n--- Best stability metrics ---")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")

    return best_metrics


# ---------------------------------------------------------------------------
# Model loading with Stage 1 transfer
# ---------------------------------------------------------------------------

def load_stage1_into_model(model, stage1_path, reinit_head, device):
    """
    Load Stage 1/2 (binding affinity / stability) weights into the model.
    Optionally reinitialize the projection head for the new task.
    Uses strict=False when the model has metadata params not in the checkpoint.
    """
    stage1_ckpt = torch_load(stage1_path, map_location=device, weights_only=False)
    state_dict = stage1_ckpt["model_state_dict"]

    if reinit_head:
        # Load only backbone weights, skip projection head
        backbone_state = {k: v for k, v in state_dict.items() if not k.startswith("project.")}
        missing = load_transfer(model, backbone_state, label="S1→S2 backbone")
        print(f"Loaded checkpoint backbone. Reinitialized head.")
        print(f"  Missing keys (expected - head): {[k for k in missing if 'project' in k]}")
    else:
        # Load everything; strict=False allows new metadata params to stay at init
        missing = load_transfer(model, state_dict, label="S1→S2 full")
        if missing:
            print(f"Loaded checkpoint. New params (randomly initialized):")
            for k in missing:
                print(f"    {k}")
        else:
            print(f"Loaded full checkpoint (backbone + head)")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Double fine-tune MINT on stability prediction")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (values in 'args' key set defaults)")

    # Data
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory containing train.csv, val.csv, test.csv for stability")
    parser.add_argument("--peptide_col", type=str, default="peptide_sequence")
    parser.add_argument("--mhc_col", type=str, default="mhc_sequence")
    parser.add_argument("--label_col", type=str, default="label")

    # Checkpoints
    parser.add_argument("--stage1_checkpoint", type=str, default=None,
                        help="Path to Stage 1 best checkpoint (best_mhc_binding.pt)")
    parser.add_argument("--mint_checkpoint", type=str, default=None,
                        help="Path to original MINT checkpoint (needed to build model arch)")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--use_multimer", action="store_true", default=False)

    # Transfer strategy
    parser.add_argument("--reinit_head", action="store_true", default=False,
                        help="Reinitialize projection head for stability task "
                             "(recommended if task_type differs from Stage 1)")
    parser.add_argument("--freeze_percent", type=float, default=0.7,
                        help="Fraction of backbone layers to freeze (higher = more frozen, "
                             "recommended higher than Stage 1 since data is smaller)")
    parser.add_argument("--freeze_stage1_head", action="store_true", default=False,
                        help="Also freeze the projection head from Stage 1 "
                             "(only when NOT reinitializing head)")

    # Model
    parser.add_argument("--hdim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)

    # Training
    parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse",
                        help="Loss function for regression (default: mse)")
    parser.add_argument("--task_type", type=str, choices=["cls", "reg"], default="reg",
                        help="cls=binary classification, reg=regression (stability score)")
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=128,
                        help="Batch size for evaluation (no gradients, can be larger)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Lower LR recommended for double fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop after N optimizer steps (0=use num_epochs instead)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--pos_weight", type=float, default=1.0)
    parser.add_argument("--early_stopping", type=int, default=10)
    parser.add_argument("--log_steps", type=int, default=10,
                        help="Log train loss every N optimizer steps")
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="Evaluate every N optimizer steps (0=epoch end only)")
    parser.add_argument("--truncation_len", type=int, default=None)
    parser.add_argument("--val_subsample", type=int, default=0,
                        help="Subsample val/test to N rows for faster sweep evals (0=use full set)")
    parser.add_argument("--log_transform", action="store_true", default=False,
                        help="Apply log(1+x) transform to labels before training. "
                             "Recommended for stability half-life data to compress the range. "
                             "Metrics will be reported in log-space (Pearson/Spearman unaffected).")
    parser.add_argument("--stability_score", action="store_true", default=False,
                        help="Transform labels from half-life hours to stability score [0,1] "
                             "via s = 2^(-1/T_half). Mutually exclusive with --log_transform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--deterministic", action="store_true",
                        help="Enable cudnn deterministic mode (slower but bit-reproducible on GPU)")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints/stage2_stability")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name (omit to disable wandb)")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = _load_args_from_config(parser)
    if args.data_dir is None:
        parser.error("--data_dir is required (via CLI or --config)")
    if args.stage1_checkpoint is None:
        parser.error("--stage1_checkpoint is required (via CLI or --config)")
    if args.mint_checkpoint is None:
        parser.error("--mint_checkpoint is required (via CLI or --config)")

    # Seed
    set_seed(args.seed, deterministic=getattr(args, "deterministic", False))

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    cfg = load_esm2_config(args.config_path)

    # Load data
    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))

    if args.val_subsample > 0:
        if args.val_subsample < len(val_df):
            val_df = val_df.sample(n=args.val_subsample, random_state=42)
        if args.val_subsample < len(test_df):
            test_df = test_df.sample(n=args.val_subsample, random_state=42)

    if args.log_transform and args.stability_score:
        parser.error("--log_transform and --stability_score are mutually exclusive")

    if args.log_transform:
        for df in [train_df, val_df, test_df]:
            df[args.label_col] = np.log1p(df[args.label_col])
        print("Applied log(1+x) transform to labels")

    if args.stability_score:
        for df in [train_df, val_df, test_df]:
            t_half = df[args.label_col].values
            # s = 2^(-1/T_half), with T_half=0 mapping to s=0
            s = np.where(t_half > 0, np.power(2.0, -1.0 / t_half), 0.0)
            df[args.label_col] = s
        print("Applied stability score transform: s = 2^(-1/T_half) -> [0, 1]")

    print(f"Stability data loaded: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Datasets (5-tuple when metadata columns provided, else standard 3-tuple)
    collate_fn = MHCCollateFn(truncation_seq_length=args.truncation_len)
    ds_kwargs = dict(
        peptide_col=args.peptide_col, mhc_col=args.mhc_col, target_col=args.label_col,
    )
    train_dataset = MHCBindingDataset(train_df, **ds_kwargs)
    val_dataset = MHCBindingDataset(val_df, **ds_kwargs)
    test_dataset = MHCBindingDataset(test_df, **ds_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=True
    )
    eval_bs = args.eval_bs if args.eval_bs > 0 else args.bs
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=eval_bs, collate_fn=collate_fn, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_bs, collate_fn=collate_fn, shuffle=False
    )

    # Build model architecture (uses MINT checkpoint to init ESM2 backbone)
    # Then overwrite with Stage 1/2 fine-tuned weights
    model = MHCBindingWrapper(
        cfg,
        args.mint_checkpoint,  # just for architecture init
        freeze_percent=args.freeze_percent,
        use_multimer=args.use_multimer,
        hidden_dim=args.hdim,
        dropout=args.dropout,
        output_size=1,
        device=args.device,
        sigmoid_output=args.stability_score,
    )

    # Load Stage 1/2 weights on top
    model = load_stage1_into_model(
        model, args.stage1_checkpoint, args.reinit_head, args.device,
    )

    # Optionally freeze the head too (if keeping Stage 1 head)
    if args.freeze_stage1_head and not args.reinit_head:
        for param in model.project.parameters():
            param.requires_grad = False
        print("Froze Stage 1 projection head")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    # Weights & Biases
    use_wandb = args.wandb_project is not None
    if use_wandb:
        if wandb is None:
            raise ImportError("wandb is required when --wandb_project is set. Install with: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    try:
        train(model, train_loader, val_loader, test_loader, cfg, args, use_wandb=use_wandb)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n*** OOM: this config exceeds GPU memory. Reporting and skipping. ***")
            torch.cuda.empty_cache()
            if use_wandb:
                wandb.log({"val_loss": 1e6, "oom": 1})
        else:
            raise

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
