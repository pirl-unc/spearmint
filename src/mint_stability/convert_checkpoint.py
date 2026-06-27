"""Convert a MINT .pt training checkpoint to HuggingFace format.

Supports all three stages:
  - s1: Stage 1 binding affinity (MintStabilityForRegression)
  - s2: Stage 2 stability (MintStabilityForRegression)
  - s3: Stage 3 SPEARMINT FiLM (SpearmintForStabilityPrediction)

Usage:
    python -m mint_stability.convert_checkpoint \
        --stage s2 \
        --checkpoint_path ../checkpoints/best_stability.pt \
        --output_dir ./mint-2stage-stability \
        --verify

    python -m mint_stability.convert_checkpoint \
        --stage s3 \
        --checkpoint_path ../checkpoints/best_stage3_film.pt \
        --output_dir ./spearmint \
        --verify

This produces:
    <output_dir>/
        config.json
        model.safetensors
        <modeling file>.py       (generated via build_hf for trust_remote_code)
        <configuration file>.py
"""

import argparse
import os
import sys

import torch

from ._compat import torch_load, load_transfer
from .configuration_mint import MintStabilityConfig
from .configuration_spearmint import SpearmintConfig
from .modeling_mint import MintStabilityForRegression
from .modeling_spearmint import SpearmintForStabilityPrediction
from .tokenizer import MintTokenizer, SpearmintTokenizer
from .build_hf import _build_monolithic
from .train_binding import MHCBindingWrapper, load_esm2_config


def _load_checked(model, state_dict, allowed_missing=frozenset()):
    """Load state_dict into model, raising on unexpected or genuinely missing keys.

    Args:
        model: nn.Module to load into.
        state_dict: Weights to load.
        allowed_missing: Set of key names that are expected to be absent
            (e.g. tied lm_head weights).

    Returns:
        The lm_head key that was allowed-missing (for logging), or None.

    Raises:
        RuntimeError: If any unexpected keys exist or any keys are missing
            beyond the allowed set.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    real_missing = [k for k in missing if k not in allowed_missing]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint (model doesn't have these "
            f"parameters): {unexpected}"
        )
    if real_missing:
        raise RuntimeError(
            f"Missing keys in checkpoint (model expects these but they're "
            f"absent): {real_missing}"
        )
    skipped = [k for k in missing if k in allowed_missing]
    if skipped:
        print(f"  Skipped expected-missing keys: {skipped}")
    else:
        print("  All keys matched perfectly.")


def convert_checkpoint(args):
    print(f"Loading checkpoint: {args.checkpoint_path}")
    ckpt = torch_load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Print checkpoint info
    if "args" in ckpt:
        ckpt_args = ckpt["args"]
        print(f"  Checkpoint args: {dict(ckpt_args) if hasattr(ckpt_args, '__iter__') else ckpt_args}")
    if "best_metrics" in ckpt:
        print(f"  Best metrics: {ckpt['best_metrics']}")
    if "epoch" in ckpt:
        print(f"  Epoch: {ckpt['epoch']}")

    # Extract checkpoint args for config inference
    ckpt_args = ckpt.get("args", {})
    if hasattr(ckpt_args, "__dict__"):
        ckpt_args = vars(ckpt_args)

    if args.stage in ("s1", "s2"):
        model, config, lm_head_key = _convert_s1_s2(args, state_dict, ckpt_args)
    elif args.stage == "s3":
        model, config, lm_head_key = _convert_s3(args, state_dict, ckpt_args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    # Save model and config
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving to {args.output_dir}/ ...")
    try:
        model.save_pretrained(args.output_dir, safe_serialization=True)
        print("  Saved model.safetensors")
    except Exception:
        model.save_pretrained(args.output_dir)
        print("  Saved pytorch_model.bin (safetensors unavailable)")
    config.save_pretrained(args.output_dir)

    # Generate self-contained HF files via build_hf
    print("\nGenerating HF-compatible modeling files...")
    modeling_source, config_source, config_filename, modeling_filename = (
        _build_monolithic(args.stage)
    )
    for fname, content in [
        (modeling_filename, modeling_source),
        (config_filename, config_source),
    ]:
        path = os.path.join(args.output_dir, fname)
        with open(path, "w") as f:
            f.write(content)
        print(f"  Wrote {fname}")

    # Print model size
    for model_file in ["model.safetensors", "pytorch_model.bin"]:
        model_path = os.path.join(args.output_dir, model_file)
        if os.path.exists(model_path):
            size_gb = os.path.getsize(model_path) / (1024 ** 3)
            print(f"  Model size ({model_file}): {size_gb:.2f} GB")
            break

    # Verify round-trip (in-memory vs saved)
    if args.verify:
        print("\n--- Round-trip verification ---")
        _verify_round_trip(args.stage, model, args.output_dir, lm_head_key)

    # Verify against original training model
    if getattr(args, "verify_original", False):
        print("\n--- Original-model verification ---")
        _verify_against_original(args, model, ckpt)

    print(f"\nDone!")


def _convert_s1_s2(args, state_dict, ckpt_args=None):
    """Convert S1 or S2 checkpoint."""
    # Infer hidden_dim from state dict
    hidden_dim = state_dict["project.0.weight"].shape[0]
    print(f"  Inferred: hidden_dim={hidden_dim}")

    # Use checkpoint-saved args as defaults, CLI overrides take precedence
    ckpt_args = ckpt_args or {}
    use_multimer = args.use_multimer if args.use_multimer is not None else ckpt_args.get("use_multimer", True)
    dropout = args.dropout if args.dropout is not None else ckpt_args.get("dropout", 0.2)
    sigmoid_output = ckpt_args.get("stability_score", False)
    token_dropout = ckpt_args.get("token_dropout", True)

    print(f"  Config: use_multimer={use_multimer}, dropout={dropout}, "
          f"sigmoid_output={sigmoid_output}, token_dropout={token_dropout}")

    # ESM2-650M architecture (the only backbone used by MINT)
    config = MintStabilityConfig(
        num_layers=33,
        embed_dim=1280,
        attention_heads=20,
        token_dropout=token_dropout,
        use_multimer=use_multimer,
        hidden_dim=hidden_dim,
        dropout=dropout,
        output_size=1,
        sigmoid_output=sigmoid_output,
    )
    config.auto_map = {
        "AutoConfig": "configuration_mint.MintStabilityConfig",
        "AutoModel": "modeling_mint_stability.MintStabilityForRegression",
    }

    print("Creating MintStabilityForRegression...")
    model = MintStabilityForRegression(config)

    lm_head_key = "model.lm_head.weight"
    _load_checked(model, state_dict, allowed_missing={lm_head_key})

    return model, config, lm_head_key


def _convert_s3(args, state_dict, ckpt_args=None):
    """Convert S3 (SPEARMINT FiLM) checkpoint."""
    # Infer dimensions from state dict
    hidden_dim = state_dict["project.0.weight"].shape[0]
    assay_emb_dim = state_dict["assay_emb.weight"].shape[1]
    num_assays = state_dict["assay_emb.weight"].shape[0]
    temp_emb_dim = state_dict["temp_linear.weight"].shape[0]
    film_hidden_dim = state_dict["film_mlp.0.weight"].shape[0]

    print(f"  Inferred: hidden_dim={hidden_dim}, assay_emb_dim={assay_emb_dim}, "
          f"num_assays={num_assays}, temp_emb_dim={temp_emb_dim}, "
          f"film_hidden_dim={film_hidden_dim}")

    # Use checkpoint-saved args as defaults, CLI overrides take precedence
    ckpt_args = ckpt_args or {}
    use_multimer = args.use_multimer if args.use_multimer is not None else ckpt_args.get("use_multimer", True)
    dropout = ckpt_args.get("dropout", 0.0)
    token_dropout = ckpt_args.get("token_dropout", True)

    print(f"  Config: use_multimer={use_multimer}, dropout={dropout}, "
          f"token_dropout={token_dropout}")

    # ESM2-650M architecture (the only backbone used by MINT)
    config = SpearmintConfig(
        num_layers=33,
        embed_dim=1280,
        attention_heads=20,
        token_dropout=token_dropout,
        use_multimer=use_multimer,
        hidden_dim=hidden_dim,
        num_assays=num_assays,
        assay_emb_dim=assay_emb_dim,
        temp_emb_dim=temp_emb_dim,
        film_hidden_dim=film_hidden_dim,
        dropout=dropout,
    )
    config.auto_map = {
        "AutoConfig": "configuration_spearmint.SpearmintConfig",
        "AutoModel": "modeling_spearmint.SpearmintForStabilityPrediction",
    }

    print("Creating SpearmintForStabilityPrediction...")
    model = SpearmintForStabilityPrediction(config)

    lm_head_key = "esm.lm_head.weight"
    _load_checked(model, state_dict, allowed_missing={lm_head_key})

    return model, config, lm_head_key


def _verify_round_trip(stage, original_model, output_dir, lm_head_key):
    """Load saved weights directly and verify via state_dict + forward pass.

    Rather than going through from_pretrained (which also uses strict=False),
    we load the saved checkpoint file directly and compare state dicts, then
    compare forward-pass outputs.
    """
    from safetensors.torch import load_file

    # 1. Direct state_dict comparison
    sf_path = os.path.join(output_dir, "model.safetensors")
    bin_path = os.path.join(output_dir, "pytorch_model.bin")

    if os.path.exists(sf_path):
        saved_sd = load_file(sf_path)
    elif os.path.exists(bin_path):
        saved_sd = torch_load(bin_path, map_location="cpu", weights_only=True)
    else:
        print("  FAIL: No saved model file found!")
        sys.exit(1)

    orig_sd = original_model.state_dict()
    # Keys that won't be in saved_sd (tied weights not saved by HF)
    allowed_missing_in_saved = {lm_head_key}

    for key in orig_sd:
        if key in allowed_missing_in_saved:
            continue
        if key not in saved_sd:
            print(f"  FAIL: Key '{key}' in original but not in saved file!")
            sys.exit(1)
        max_param_diff = (orig_sd[key].float() - saved_sd[key].float()).abs().max().item()
        if max_param_diff > 1e-6:
            print(f"  FAIL: Parameter '{key}' differs by {max_param_diff:.2e}")
            sys.exit(1)

    extra_saved = set(saved_sd.keys()) - set(orig_sd.keys())
    if extra_saved:
        print(f"  FAIL: Saved file has extra keys: {extra_saved}")
        sys.exit(1)

    print("  State dict comparison: PASS")

    # 2. Forward-pass comparison
    original_model.eval()
    tokenizer = MintTokenizer() if stage in ("s1", "s2") else SpearmintTokenizer()
    peptide = "GILGFVFTL"
    mhc_seq = "MAVMAPRTLLLLLSGALALTQTWAG"

    if stage in ("s1", "s2"):
        chains, chain_ids = tokenizer.prepare_input(peptide, mhc_seq)
        chains = chains.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)
        with torch.no_grad():
            orig_out = original_model(chains, chain_ids)["logits"]
    else:
        chains, chain_ids, assay_idxs, temp_floats = tokenizer.prepare_input(
            peptide, mhc_seq, assay="SPA", temperature_c=37.0,
        )
        chains = chains.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)
        with torch.no_grad():
            orig_out = original_model(
                chains, chain_ids, assay_idxs, temp_floats
            )["logits"]

    # Reload from saved weights into a fresh model and compare
    if stage in ("s1", "s2"):
        config = MintStabilityConfig.from_pretrained(output_dir)
        loaded_model = MintStabilityForRegression(config)
        loaded_lm_key = "model.lm_head.weight"
    else:
        config = SpearmintConfig.from_pretrained(output_dir)
        loaded_model = SpearmintForStabilityPrediction(config)
        loaded_lm_key = "esm.lm_head.weight"

    _load_checked(loaded_model, saved_sd, allowed_missing={loaded_lm_key})
    loaded_model.eval()

    with torch.no_grad():
        if stage in ("s1", "s2"):
            loaded_out = loaded_model(chains, chain_ids)["logits"]
        else:
            loaded_out = loaded_model(
                chains, chain_ids, assay_idxs, temp_floats
            )["logits"]

    max_diff = (orig_out - loaded_out).abs().max().item()
    print(f"  Original output:  {orig_out.item():.6f}")
    print(f"  Loaded output:    {loaded_out.item():.6f}")
    print(f"  Max abs diff:     {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  PASS: Round-trip verified.")
    else:
        print("  FAIL: Outputs differ!")
        sys.exit(1)


def _verify_against_original(args, hf_model, ckpt):
    """Verify converted HF model against the original training-time model.

    Rebuilds the original model exactly as training code does (MHCBindingWrapper
    for S1/S2, Stage3FiLM wrapping MHCBindingWrapper for S3), loads checkpoint
    weights, runs a forward pass, and asserts the HF model matches.

    Requires:
        --mint_checkpoint: path to the original MINT backbone (.ckpt)
        --stage2_checkpoint: path to S2 .pt (only for S3 stage)
    """
    ckpt_args = ckpt.get("args", {})
    if hasattr(ckpt_args, "__dict__"):
        ckpt_args = vars(ckpt_args)

    # Resolve MINT backbone path
    mint_ckpt = args.mint_checkpoint
    if mint_ckpt is None:
        mint_ckpt = ckpt_args.get("checkpoint_path") or ckpt_args.get("mint_checkpoint")
    if mint_ckpt is None or not os.path.exists(mint_ckpt):
        print("  FAIL: --verify_original requires --mint_checkpoint (or a "
              "checkpoint_path stored in the checkpoint args). Cannot proceed.")
        sys.exit(1)

    cfg = load_esm2_config()

    tokenizer = MintTokenizer() if args.stage in ("s1", "s2") else SpearmintTokenizer()
    peptide = "GILGFVFTL"
    mhc_seq = "MAVMAPRTLLLLLSGALALTQTWAG"

    if args.stage in ("s1", "s2"):
        # Rebuild original S1/S2 model
        use_multimer = ckpt_args.get("use_multimer", True)
        hdim = ckpt_args.get("hdim", 512)
        dropout = ckpt_args.get("dropout", 0.2)
        sigmoid_output = ckpt_args.get("stability_score", False)

        orig = MHCBindingWrapper(
            cfg, mint_ckpt,
            freeze_percent=0.0,
            use_multimer=use_multimer,
            hidden_dim=hdim,
            dropout=dropout,
            output_size=1,
            device="cpu",
            sigmoid_output=sigmoid_output,
        )
        state_dict = ckpt["model_state_dict"]
        load_transfer(orig, state_dict, label="verify S1/S2 original")
        orig.eval()

        chains, chain_ids = tokenizer.prepare_input(peptide, mhc_seq)
        chains = chains.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        with torch.no_grad():
            orig_out = orig(chains, chain_ids)
            hf_out = hf_model(chains, chain_ids)["logits"]

    else:
        # S3: need Stage 2 checkpoint to build base model
        s2_path = args.stage2_checkpoint
        if s2_path is None:
            print("  FAIL: --verify_original for S3 requires --stage2_checkpoint.")
            sys.exit(1)
        if not os.path.exists(s2_path):
            print(f"  FAIL: stage2_checkpoint not found: {s2_path}")
            sys.exit(1)

        from .train_stage3 import Stage3FiLM

        s2_ckpt = torch_load(s2_path, map_location="cpu", weights_only=False)
        s2_args = s2_ckpt.get("args", {})
        if hasattr(s2_args, "__dict__"):
            s2_args = vars(s2_args)

        use_multimer = s2_args.get("use_multimer", True)
        hdim = s2_args.get("hdim", 512)
        s2_dropout = s2_args.get("dropout", 0.2)
        freeze_percent = s2_args.get("freeze_percent", 0.7)

        base_model = MHCBindingWrapper(
            cfg, mint_ckpt,
            freeze_percent=freeze_percent,
            use_multimer=use_multimer,
            hidden_dim=hdim,
            dropout=s2_dropout,
            output_size=1,
            device="cpu",
            sigmoid_output=False,
        )
        load_transfer(base_model, s2_ckpt["model_state_dict"], label="verify S3 base (S2)")

        # Infer S3 FiLM args from checkpoint
        s3_state = ckpt["model_state_dict"]
        num_assays = s3_state["assay_emb.weight"].shape[0]
        assay_emb_dim = s3_state["assay_emb.weight"].shape[1]
        temp_emb_dim = s3_state["temp_linear.weight"].shape[0]
        film_hidden_dim = s3_state["film_mlp.0.weight"].shape[0]
        s3_dropout = ckpt_args.get("dropout", 0.0)

        orig = Stage3FiLM(
            base_model,
            num_assays=num_assays,
            assay_emb_dim=assay_emb_dim,
            temp_emb_dim=temp_emb_dim,
            film_hidden_dim=film_hidden_dim,
            dropout=s3_dropout,
            unfreeze_project=False,
        )
        load_transfer(orig, s3_state, label="verify S3 FiLM")
        orig.eval()

        chains, chain_ids, assay_idxs, temp_floats = tokenizer.prepare_input(
            peptide, mhc_seq, assay="SPA", temperature_c=37.0,
        )
        chains = chains.unsqueeze(0)
        chain_ids = chain_ids.unsqueeze(0)

        with torch.no_grad():
            orig_out = orig(chains, chain_ids, assay_idxs, temp_floats)
            if orig_out.dim() == 1:
                orig_out = orig_out.unsqueeze(-1)
            hf_out = hf_model(
                chains, chain_ids, assay_idxs, temp_floats,
            )["logits"]

    max_diff = (orig_out - hf_out).abs().max().item()
    print(f"  Original model output: {orig_out.item():.6f}")
    print(f"  HF model output:       {hf_out.item():.6f}")
    print(f"  Max abs diff:           {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  PASS: Original vs HF verified.")
    else:
        print("  FAIL: Outputs differ beyond tolerance!")
        sys.exit(1)


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MINT .pt checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--stage", type=str, required=True, choices=["s1", "s2", "s3"],
        help="Model stage: s1, s2, or s3",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory for HF model files",
    )
    parser.add_argument(
        "--use_multimer", type=_parse_bool, default=None,
        help="Whether model uses multimer attention (inferred from checkpoint if not set)",
    )
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="Projection head dropout (inferred from checkpoint if not set)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify round-trip: save -> load -> same output",
    )
    parser.add_argument(
        "--verify_original", action="store_true",
        help="Verify HF model matches original training model output",
    )
    parser.add_argument(
        "--mint_checkpoint", type=str, default=None,
        help="Path to original MINT backbone (.ckpt). "
             "Inferred from checkpoint args if not set.",
    )
    parser.add_argument(
        "--stage2_checkpoint", type=str, default=None,
        help="Path to S2 .pt checkpoint (required for S3 --verify_original)",
    )
    args = parser.parse_args()
    convert_checkpoint(args)
