"""Utility for loading YAML configuration files with `_base` inheritance."""

from __future__ import annotations
import copy
import os
import yaml
from typing import Any, Dict


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` into `base`. `override` wins on leaves."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(val, dict)
        ):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config, resolving `_base` inheritance if present.

    If the loaded YAML contains a top-level `_base` key (string path,
    relative to the config file's own directory, or absolute), the base
    file is loaded first and the current file is deep-merged on top.

    The `_base` key is removed from the returned dictionary.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Parsed YAML contents as a nested dictionary, with inheritance
        resolved.

    Raises:
        FileNotFoundError: If the file (or its `_base`) does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base", None)
    if base_ref is None:
        return raw

    if not os.path.isabs(base_ref):
        base_ref = os.path.join(os.path.dirname(path), base_ref)
    base_cfg = load_yaml(base_ref)
    return _deep_merge(base_cfg, raw)
