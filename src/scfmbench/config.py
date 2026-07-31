"""Config loading. Every analysis constant comes from a YAML file, never a
literal in a script (repository requirement: configuration-driven)."""
from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml

_ENV_PREFIX = "SCFM_"


class ConfigError(RuntimeError):
    """Raised on a missing or malformed config. Fail loudly, never default silently."""


def _expand(node: Any) -> Any:
    """Recursively expand ${VAR} against the environment.

    A referenced-but-unset variable is an error, not an empty string -- an
    empty data root would silently write artefacts into the CWD.
    """
    if isinstance(node, str):
        out = os.path.expandvars(node)
        if "${" in out:
            missing = out[out.index("${") + 2 : out.index("}")]
            raise ConfigError(
                f"config references environment variable {missing!r} which is not set. "
                f"Set it (e.g. export {_ENV_PREFIX}DATA_ROOT=/path/to/volume) and retry."
            )
        return out
    if isinstance(node, dict):
        return {k: _expand(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand(v) for v in node]
    return node


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | pathlib.Path) -> dict:
    """Load a config, resolving a single level of `inherit:` and expanding ${VARS}."""
    path = pathlib.Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(load(path.parent.parent / parent if not pathlib.Path(parent).exists() else parent), cfg)
    return _expand(cfg)
