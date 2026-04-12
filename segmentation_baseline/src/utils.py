"""
General utilities: config loading, logging setup, path resolution.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger for CLI scripts."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def resolve_path(base: Path, rel: str) -> Path:
    """Resolve a path relative to a base directory."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
