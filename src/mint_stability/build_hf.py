"""Build self-contained HuggingFace modeling files by inlining the backbone.

HuggingFace's trust_remote_code=True requires that each repo's modeling file
is self-contained (no cross-file imports beyond the config). This script reads
the factored source files from mint_stability/ and assembles them into a single
monolithic modeling file suitable for upload to HuggingFace Hub.

Usage:
    # Build S1/S2 (MINT stability) HF files:
    python -m mint_stability.build_hf --stage s2 --output_dir ./hf-mint-stability

    # Build S3 (SPEARMINT) HF files:
    python -m mint_stability.build_hf --stage s3 --output_dir ./hf-spearmint

    # Verify the generated file works:
    python -m mint_stability.build_hf --stage s2 --output_dir ./hf-test --verify
"""

import argparse
import ast
import importlib
import os
import sys
import tempfile


# ============================================================================
# AST-based source processor
# ============================================================================

# Import categories used by strip_body()
_CAT_STDLIB = "stdlib"
_CAT_TYPING = "typing"
_CAT_TORCH = "torch"
_CAT_TRANSFORMERS = "transformers"
_CAT_LOCAL = "local_relative"


class _ASTSourceProcessor:
    """Analyse Python source via AST, strip/extract by line ranges.

    Uses ``ast.parse`` to classify import statements and locate class
    definitions, but outputs *original source lines* — preserving all
    comments, formatting and whitespace.  Compatible with Python 3.7
    (no ``end_lineno`` or ``ast.unparse``).
    """

    def __init__(self, source, filename="<string>"):
        self.source = source
        self.lines = source.split("\n")
        self.tree = ast.parse(source, filename)

    # ------------------------------------------------------------------
    # Import classification
    # ------------------------------------------------------------------

    def _classify_import(self, node):
        """Return a category string for a top-level Import/ImportFrom node."""
        if isinstance(node, ast.Import):
            name = node.names[0].name.split(".")[0]
            if name in ("torch",):
                return _CAT_TORCH
            return _CAT_STDLIB
        if isinstance(node, ast.ImportFrom):
            level = node.level  # >0 means relative
            if level > 0:
                return _CAT_LOCAL
            module = node.module or ""
            root = module.split(".")[0]
            if root == "typing":
                return _CAT_TYPING
            if root in ("torch",):
                return _CAT_TORCH
            if root in ("transformers",):
                return _CAT_TRANSFORMERS
            return _CAT_STDLIB
        return None

    # ------------------------------------------------------------------
    # Line-range helpers (Python 3.7 safe — no end_lineno)
    # ------------------------------------------------------------------

    @staticmethod
    def _import_end(lines, start):
        """Return the last line index of an import starting at *start*."""
        line = lines[start]
        # Multi-line import with parens
        if "(" in line and ")" not in line:
            i = start + 1
            while i < len(lines):
                if ")" in lines[i]:
                    return i
                i += 1
        # Backslash continuation
        if line.rstrip().endswith("\\"):
            i = start + 1
            while i < len(lines) and lines[i - 1].rstrip().endswith("\\"):
                i += 1
            return i - 1
        return start

    @staticmethod
    def _block_end(lines, start):
        """Given a line starting an indented block, find its last line."""
        i = start + 1
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped and not lines[i][0].isspace() and not stripped.startswith("#"):
                break
            i += 1
        # Trim trailing blanks
        end = i - 1
        while end > start and not lines[end].strip():
            end -= 1
        return end

    # ------------------------------------------------------------------
    # Module docstring / __all__
    # ------------------------------------------------------------------

    def _docstring_range(self):
        """Return (start, end) of the module docstring, or None."""
        if not self.tree.body:
            return None
        first = self.tree.body[0]
        # Python 3.7 uses ast.Str; 3.8+ uses ast.Constant
        is_str = (isinstance(first, ast.Expr)
                  and (isinstance(first.value, ast.Str)
                       or (isinstance(first.value, ast.Constant)
                           and isinstance(first.value.value, str))))
        if not is_str:
            return None
        # Python 3.8+ (package requires >=3.9): lineno is the FIRST line
        # and end_lineno the LAST -- use them directly.
        end_lineno = getattr(first, "end_lineno", None)
        if end_lineno is not None:
            return (first.lineno - 1, end_lineno - 1)
        # Python 3.7 fallback: lineno points to the LAST line of the string.
        # In 3.8+, it points to the first line. We scan backwards from
        # lineno to find the opening triple-quote.
        end = first.lineno - 1
        start = end
        for i in range(end - 1, -1, -1):
            if '"""' in self.lines[i] or "'''" in self.lines[i]:
                start = i
                break
        return (start, end)

    def _dunder_all_range(self):
        """Return (start, end) of ``__all__ = [...]``, or None."""
        for node in ast.iter_child_nodes(self.tree):
            if (isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__all__"):
                start = node.lineno - 1
                end = self._import_end(self.lines, start)
                return (start, end)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def strip_body(self, categories):
        """Remove module docstring, ``__all__``, and imports in *categories*.

        Args:
            categories: set of category strings (e.g. ``{"stdlib", "torch"}``).

        Returns:
            Cleaned source string.
        """
        remove = set()

        # Docstring
        ds = self._docstring_range()
        if ds:
            remove.update(range(ds[0], ds[1] + 1))

        # __all__
        da = self._dunder_all_range()
        if da:
            remove.update(range(da[0], da[1] + 1))

        # Imports
        for node in ast.iter_child_nodes(self.tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            cat = self._classify_import(node)
            if cat in categories:
                start = node.lineno - 1
                end = self._import_end(self.lines, start)
                remove.update(range(start, end + 1))

        result = [l for i, l in enumerate(self.lines) if i not in remove]
        return "\n".join(result)

    def extract_class(self, class_name):
        """Extract a class definition by name, returning its source."""
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                start = node.lineno - 1
                # Include any decorators
                if node.decorator_list:
                    start = node.decorator_list[0].lineno - 1
                end = self._block_end(self.lines, node.lineno - 1)
                return "\n".join(self.lines[start:end + 1])
        raise ValueError(f"Class {class_name} not found in source")

    def find_external_deps(self, class_name):
        """Find imported names referenced inside a class body.

        Returns:
            dict mapping local_name -> (original_name, source_module)
            for each module-level relative import used by the class.
        """
        # 1. Collect module-level relative imports: local_name -> (original, module)
        imported = {}
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                mod = node.module or ""
                for alias in node.names:
                    local = alias.asname or alias.name
                    imported[local] = (alias.name, mod)

        # 2. Collect all Name references inside the class
        class_node = None
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                class_node = node
                break
        if class_node is None:
            raise ValueError(f"Class {class_name} not found in source")

        used = set()
        for node in ast.walk(class_node):
            if isinstance(node, ast.Name):
                used.add(node.id)

        return {name: imported[name] for name in used if name in imported}


# ============================================================================
# Categories to strip per source file
# ============================================================================

# All inlined files: strip their stdlib, typing, torch, transformers, and
# relative imports — the header already provides these.
_INLINE_CATEGORIES = {_CAT_STDLIB, _CAT_TYPING, _CAT_TORCH,
                      _CAT_TRANSFORMERS, _CAT_LOCAL}


# ============================================================================
# File reading
# ============================================================================

def _src_dir():
    """Return the directory containing this script (mint_stability package)."""
    return os.path.dirname(os.path.abspath(__file__))


def _read_file(filename):
    """Read a source file from the mint_stability package."""
    path = os.path.join(_src_dir(), filename)
    with open(path, "r") as f:
        return f.read()


# ============================================================================
# Header builder
# ============================================================================

def _build_header(config_module, config_class, extra_config_imports=None):
    """Build the file header with imports and config import fallback.

    Args:
        config_module: Name of the config module (e.g. "configuration_spearmint").
        config_class: Name of the config class (e.g. "SpearmintConfig").
        extra_config_imports: Optional list of additional names to import from
            the config module (e.g. ["DEFAULT_ASSAY_TYPES", "DEFAULT_TEMP_C"]).
    """
    all_imports = [config_class]
    if extra_config_imports:
        all_imports.extend(extra_config_imports)
    import_names = ", ".join(all_imports)
    fallback_bindings = "\n".join("    %s = _cfg.%s" % (n, n) for n in all_imports)

    return f'''"""
Self-contained model file for HuggingFace Hub (trust_remote_code=True).

Inlines all dependencies from the MINT package (Ullanat et al., 2026) so that
users can load the model without installing mint:

    from transformers import AutoModel
    model = AutoModel.from_pretrained("...", trust_remote_code=True)

Original MINT source: https://github.com/VarunUllanat/mint (MIT License)
ESM2 backbone: Meta Platforms, Inc. (MIT License)

Generated by: python -m mint_stability.build_hf
"""

import inspect
import itertools
import math
import os
import uuid
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from transformers import PreTrainedModel

try:
    from .{config_module} import {import_names}
except ImportError:  # standalone / direct import (no package context)
    import importlib as _il
    _cfg = _il.import_module("{config_module}")
{fallback_bindings}


# --- torch.load compat (inlined from _compat.py) ---
_TORCH_LOAD_PARAMS = set(inspect.signature(torch.load).parameters)

def torch_load(f, **kwargs):
    if "weights_only" not in _TORCH_LOAD_PARAMS:
        kwargs.pop("weights_only", None)
    return torch.load(f, **kwargs)

'''


# ============================================================================
# Monolithic file assembly
# ============================================================================

def _build_monolithic(stage):
    """Assemble a self-contained modeling file for the given stage.

    Args:
        stage: "s1", "s2", or "s3"

    Returns:
        tuple: (modeling_source, config_source, config_filename, modeling_filename)
    """
    # Read & strip backbone
    backbone_proc = _ASTSourceProcessor(_read_file("backbone.py"), "backbone.py")
    backbone_body = backbone_proc.strip_body(_INLINE_CATEGORIES)

    # Read & strip modeling_base
    base_proc = _ASTSourceProcessor(_read_file("modeling_base.py"), "modeling_base.py")
    base_body = base_proc.strip_body(_INLINE_CATEGORIES)

    if stage in ("s1", "s2"):
        config_module = "configuration_mint"
        config_class = "MintStabilityConfig"
        config_filename = f"{config_module}.py"
        modeling_filename = "modeling_mint_stability.py"

        # Read & strip model source
        model_proc = _ASTSourceProcessor(
            _read_file("modeling_mint.py"), "modeling_mint.py")
        model_body = model_proc.strip_body(_INLINE_CATEGORIES)

        # Read tokenizer — extract only MintTokenizer class
        tok_proc = _ASTSourceProcessor(
            _read_file("tokenizer.py"), "tokenizer.py")
        tokenizer_body = tok_proc.extract_class("MintTokenizer")

    elif stage == "s3":
        config_module = "configuration_spearmint"
        config_class = "SpearmintConfig"
        config_filename = f"{config_module}.py"
        modeling_filename = "modeling_spearmint.py"

        # Read & strip model source
        model_proc = _ASTSourceProcessor(
            _read_file("modeling_spearmint.py"), "modeling_spearmint.py")
        model_body = model_proc.strip_body(_INLINE_CATEGORIES)

        # Read tokenizer — extract only SpearmintTokenizer class
        tok_proc = _ASTSourceProcessor(
            _read_file("tokenizer.py"), "tokenizer.py")
        tokenizer_body = tok_proc.extract_class("SpearmintTokenizer")

    else:
        raise ValueError(f"Unknown stage: {stage}. Expected 's1', 's2', or 's3'.")

    # --- Auto-detect config dependencies from extracted tokenizer class ---
    # Only names imported from the config module need to be added to the
    # header import.  Names from other inlined modules (e.g. backbone) are
    # already present in the monolithic file.
    extra_imports = []
    tokenizer_preamble = ""
    class_name = "MintTokenizer" if stage in ("s1", "s2") else "SpearmintTokenizer"
    config_module_short = config_module.replace("configuration_", "")
    deps = tok_proc.find_external_deps(class_name)
    for local_name, (original_name, dep_mod) in sorted(deps.items()):
        if dep_mod != config_module_short and dep_mod != config_module:
            continue  # from an inlined module — already present
        if original_name not in extra_imports:
            extra_imports.append(original_name)
        if local_name != original_name:
            tokenizer_preamble += f"{local_name} = {original_name}\n"
    if tokenizer_preamble:
        tokenizer_preamble += "\n"

    header = _build_header(
        config_module, config_class,
        extra_imports if extra_imports else None,
    )

    sections = [
        header,
        "# " + "=" * 76,
        "# Backbone (inlined from mint_stability/backbone.py)",
        "# " + "=" * 76,
        "",
        backbone_body.strip(),
        "",
        "",
        "# " + "=" * 76,
        "# Base PreTrainedModel (inlined from mint_stability/modeling_base.py)",
        "# " + "=" * 76,
        "",
        base_body.strip(),
        "",
        "",
        "# " + "=" * 76,
        "# HuggingFace PreTrainedModel wrapper",
        "# " + "=" * 76,
        "",
        model_body.strip(),
        "",
        "",
        "# " + "=" * 76,
        "# Tokenizer helper",
        "# " + "=" * 76,
        "",
        "",
        tokenizer_preamble + tokenizer_body.strip(),
        "",
    ]

    modeling_source = "\n".join(sections)

    # Read config source (used as-is, no transformation needed)
    config_source = _read_file(config_filename)

    return modeling_source, config_source, config_filename, modeling_filename


# ============================================================================
# Legacy API (kept for backward compatibility with tests)
# ============================================================================

def _extract_class(source, class_name):
    """Extract a single class definition from source code."""
    proc = _ASTSourceProcessor(source, "<string>")
    return proc.extract_class(class_name)


# ============================================================================
# Build & verify CLI
# ============================================================================

def build(args):
    """Build self-contained HF files for the given stage."""
    modeling_source, config_source, config_filename, modeling_filename = (
        _build_monolithic(args.stage)
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Write modeling file
    modeling_path = os.path.join(args.output_dir, modeling_filename)
    with open(modeling_path, "w") as f:
        f.write(modeling_source)
    print(f"  Wrote {modeling_filename} ({len(modeling_source)} bytes)")

    # Write config file
    config_path = os.path.join(args.output_dir, config_filename)
    with open(config_path, "w") as f:
        f.write(config_source)
    print(f"  Wrote {config_filename} ({len(config_source)} bytes)")

    # Verify the generated file can be imported and instantiated
    if args.verify:
        print("\n--- Verification ---")
        _verify(args.output_dir, args.stage, modeling_filename, config_filename)

    print(f"\nDone! HF files written to {args.output_dir}/")


def _verify(output_dir, stage, modeling_filename, config_filename):
    """Verify the generated monolithic file works correctly."""
    import torch

    # Add output dir to sys.path for import
    sys.path.insert(0, output_dir)

    try:
        # Import the generated config module
        config_mod_name = config_filename.replace(".py", "")
        modeling_mod_name = modeling_filename.replace(".py", "")

        # Force reimport if already loaded
        for mod_name in [config_mod_name, modeling_mod_name]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]

        config_mod = importlib.import_module(config_mod_name)
        modeling_mod = importlib.import_module(modeling_mod_name)

        if stage in ("s1", "s2"):
            ConfigCls = getattr(config_mod, "MintStabilityConfig")
            ModelCls = getattr(modeling_mod, "MintStabilityForRegression")
            TokenizerCls = getattr(modeling_mod, "MintTokenizer")

            config = ConfigCls()
            model = ModelCls(config)
            model.eval()

            tokenizer = TokenizerCls()
            chains, chain_ids = tokenizer.prepare_input("GILGFVFTL", "MAVMAPRTL")
            with torch.no_grad():
                out = model(chains.unsqueeze(0), chain_ids.unsqueeze(0))
            print(f"  Model output shape: {out['logits'].shape}")
            print(f"  Sample output: {out['logits'].item():.6f}")

        elif stage == "s3":
            ConfigCls = getattr(config_mod, "SpearmintConfig")
            ModelCls = getattr(modeling_mod, "SpearmintForStabilityPrediction")
            TokenizerCls = getattr(modeling_mod, "SpearmintTokenizer")

            config = ConfigCls()
            model = ModelCls(config)
            model.eval()

            tokenizer = TokenizerCls()
            chains, chain_ids, assay_idx, temp = tokenizer.prepare_input(
                "GILGFVFTL", "MAVMAPRTL", assay="SPA", temperature_c=37.0
            )
            with torch.no_grad():
                out = model(
                    chains.unsqueeze(0), chain_ids.unsqueeze(0),
                    assay_idx, temp,
                )
            print(f"  Model output shape: {out['logits'].shape}")
            print(f"  Sample output: {out['logits'].item():.6f}")

        print("  PASS: Generated file imports and runs correctly.")

    finally:
        sys.path.pop(0)
        # Clean up imported modules
        for mod_name in [config_mod_name, modeling_mod_name]:
            if mod_name in sys.modules:
                del sys.modules[mod_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build self-contained HF modeling files from mint_stability package"
    )
    parser.add_argument(
        "--stage", type=str, required=True, choices=["s1", "s2", "s3"],
        help="Model stage: s1 (binding), s2 (stability), s3 (SPEARMINT FiLM)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to write the generated HF files",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the generated file can be imported and run",
    )
    args = parser.parse_args()
    build(args)
