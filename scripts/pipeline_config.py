"""Single-source loader for the pipeline's tunable parameters.

Reads `pipeline_config.env` (plain ``KEY=value``) and exposes typed constants
that the pipeline's Python scripts import instead of hardcoding magic numbers.

Resolution order for the config file:
    1. ``$PIPELINE_CONFIG`` (explicit path), if it points at a real file
    2. the first ``pipeline_config.env`` found walking up from this file
       (scripts dir -> repo root)

For each individual key, a value already present in ``os.environ`` wins over
the file. That keeps things consistent when ``run_pipeline.sh`` sources the
same file with ``set -a`` (exporting every key) before launching Python, while
still letting the scripts run standalone by parsing the file directly.

Every getter has a hardcoded fallback equal to the script's original value, so
a missing/partial config never changes behaviour — it just reproduces the old
defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

CONFIG_FILENAME = "pipeline_config.env"


def _find_config_file() -> Optional[Path]:
    explicit = os.environ.get("PIPELINE_CONFIG")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p

    here = Path(__file__).resolve()
    for directory in (here.parent, *here.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def _parse_env_file(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


_CONFIG_PATH = _find_config_file()
_FILE_VALUES: Dict[str, str] = _parse_env_file(_CONFIG_PATH) if _CONFIG_PATH else {}


def _raw(key: str) -> Optional[str]:
    # Process environment (e.g. exported by run_pipeline.sh) takes precedence.
    if key in os.environ and os.environ[key] != "":
        return os.environ[key]
    return _FILE_VALUES.get(key)


def get_str(key: str, default: str) -> str:
    v = _raw(key)
    return v if v is not None else default


def get_float(key: str, default: float) -> float:
    v = _raw(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def get_int(key: str, default: int) -> int:
    v = _raw(key)
    if v is None:
        return default
    try:
        return int(float(v))
    except ValueError:
        return default


# --- Typed parameters (fallbacks == original hardcoded values) ---------------

OUT_GROUND_DIR = get_str("OUT_GROUND_DIR", "./out_ground")
OUT_CLUSTER_DIR = get_str("OUT_CLUSTER_DIR", "./out_cluster")
OUT_CLUSTER_LAS_DIR = get_str("OUT_CLUSTER_LAS_DIR", "./out_cluster_las")

SMRF_PARAMS: Dict[str, float] = {
    "slope": get_float("SMRF_SLOPE", 0.15),
    "window": get_float("SMRF_WINDOW", 16.0),
    "threshold": get_float("SMRF_THRESHOLD", 0.5),
    "scalar": get_float("SMRF_SCALAR", 1.25),
}

CLUSTER_TOLERANCE = get_float("CLUSTER_TOLERANCE", 0.4)
MIN_CLUSTER_SIZE = get_int("MIN_CLUSTER_SIZE", 50000)
MAX_CLUSTER_SIZE = get_int("MAX_CLUSTER_SIZE", 850000)
MAX_VALID_CLUSTERS = get_int("MAX_VALID_CLUSTERS", 50)

NDVI_EPS = get_float("NDVI_EPS", 1e-6)
NDVI_LOW_THRESHOLD = get_float("NDVI_LOW_THRESHOLD", 0.2)

VOXEL_AUTO_FACTOR = get_float("VOXEL_AUTO_FACTOR", 0.05)
VOXEL_AUTO_MIN = get_float("VOXEL_AUTO_MIN", 0.01)
VOXEL_AUTO_MAX = get_float("VOXEL_AUTO_MAX", 0.20)

SLICE_N_SLICES = get_int("SLICE_N_SLICES", 30)

RANSAC_LOW_FRACTION = get_float("RANSAC_LOW_FRACTION", 0.10)
RANSAC_N_ITER = get_int("RANSAC_N_ITER", 200)
RANSAC_DIST_THRESH = get_float("RANSAC_DIST_THRESH", 0.10)
