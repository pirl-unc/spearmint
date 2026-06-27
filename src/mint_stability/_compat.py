"""Cross-version compatibility helpers."""

import inspect

import torch

__all__ = ["torch_load", "load_transfer"]


def torch_load(f, **kwargs):
    """``torch.load`` with ``weights_only`` compat for torch < 1.13.

    PyTorch 1.13 added the ``weights_only`` parameter.  Passing it on
    older versions raises a ``TypeError``.  This wrapper inspects the
    signature at import time and silently drops the kwarg when it is not
    supported, so callers can always pass ``weights_only=`` explicitly.
    """
    if "weights_only" not in _TORCH_LOAD_PARAMS:
        kwargs.pop("weights_only", None)
    return torch.load(f, **kwargs)


_TORCH_LOAD_PARAMS = set(inspect.signature(torch.load).parameters)


def load_transfer(model, state_dict, label=""):
    """``load_state_dict`` for transfer learning: allow missing, reject unexpected.

    During transfer (e.g. S1→S2 or S2→S3), missing keys are expected
    (new head parameters, metadata, etc.) but unexpected keys signal a
    checkpoint mismatch and should fail loudly.

    Returns:
        List of missing key names.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in {label} checkpoint: {unexpected}"
        )
    return missing
