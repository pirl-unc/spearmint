"""Shared base class for MINT/SPEARMINT HuggingFace PreTrainedModel wrappers."""

import os

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from ._compat import torch_load

__all__ = ["StabilityPreTrainedModel"]


class StabilityPreTrainedModel(PreTrainedModel):
    """Shared base for MINT and SPEARMINT stability models.

    Provides:
      - Common weight initialization (_init_weights)
      - Robust from_pretrained() that works across transformers versions
    """

    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Default weight init -- overridden by from_pretrained() loading."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Load model with explicit state_dict loading.

        The default PreTrainedModel.from_pretrained() loading pipeline varies
        significantly across transformers versions (meta-device init, key
        prefix stripping, tied-weight handling, _fast_init, etc.).  This
        override sidesteps all of that and just does a straightforward
        ``load_state_dict`` which works identically everywhere.
        """
        # --- resolve to a local directory --------------------------------
        if os.path.isdir(pretrained_model_name_or_path):
            model_dir = pretrained_model_name_or_path
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=kwargs.get("cache_dir"),
                token=kwargs.get("token") or kwargs.get("use_auth_token"),
                revision=kwargs.get("revision"),
            )

        # --- config -------------------------------------------------------
        config = cls.config_class.from_pretrained(model_dir)

        # --- build model (random init, post_init runs _init_weights) ------
        model = cls(config)

        # --- load checkpoint weights --------------------------------------
        sf_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")

        if os.path.exists(sf_path):
            from safetensors.torch import load_file

            state_dict = load_file(sf_path)
        elif os.path.exists(bin_path):
            state_dict = torch_load(bin_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(
                f"No model weights found in {model_dir}"
            )

        # Use _keys_to_ignore_on_load_missing from the subclass
        allowed_missing = set(getattr(cls, "_keys_to_ignore_on_load_missing", []))

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        real_missing = [k for k in missing if k not in allowed_missing]
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {unexpected}"
            )
        if real_missing:
            raise RuntimeError(
                f"Missing keys in checkpoint: {real_missing}"
            )

        # --- optional: move to device / dtype ----------------------------
        device_map = kwargs.get("device_map")
        if device_map == "auto" or device_map == "cuda":
            model = model.cuda()
        elif device_map is not None:
            import warnings
            warnings.warn(
                f"device_map={device_map!r} is not supported; ignoring. "
                "Use 'auto' or 'cuda'.",
                stacklevel=2,
            )
        dtype = kwargs.get("torch_dtype")
        if dtype is not None:
            model = model.to(dtype=dtype)

        model.eval()
        return model
