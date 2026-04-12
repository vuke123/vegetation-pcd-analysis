"""
Per-point feature extraction and selection.

Builds the feature matrix that goes into the network, based on the
configured feature_mode or feature_list.  All feature modes include XYZ;
additional channels are appended in a fixed, documented order.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Feature modes map to ordered lists of dimension names (beyond xyz).
FEATURE_MODES: Dict[str, List[str]] = {
    "xyz": [],
    "xyz_ndvi": ["ndvi"],
    "xyz_rgb": ["red", "green", "blue"],
    "xyz_rgb_nir": ["red", "green", "blue", "nir"],
    "xyz_rgb_nir_ndvi_intensity": ["red", "green", "blue", "nir", "ndvi", "intensity"],
}

# NIR can appear under different names in LAS files.
NIR_ALIASES = {"nir", "infrared", "near_infrared"}


def resolve_feature_list(cfg: dict) -> List[str]:
    """
    Return the ordered list of extra feature names (beyond xyz) from config.

    Looks for cfg['feature_list'] first; falls back to cfg['feature_mode'].
    """
    if "feature_list" in cfg and cfg["feature_list"]:
        names = list(cfg["feature_list"])
        # Strip xyz if the user included them — we always prepend xyz.
        return [n for n in names if n not in ("x", "y", "z")]

    mode = cfg.get("feature_mode", "xyz")
    if mode not in FEATURE_MODES:
        raise ValueError(
            f"Unknown feature_mode: {mode!r}. "
            f"Available: {list(FEATURE_MODES)}"
        )
    return FEATURE_MODES[mode]


def _resolve_nir(dims: dict) -> np.ndarray | None:
    """Find the NIR channel under any known alias."""
    for alias in NIR_ALIASES:
        if alias in dims:
            return dims[alias]
    return None


def extract_features(xyz: np.ndarray,
                     dims: dict,
                     feature_names: List[str],
                     normalize_xyz: bool = True) -> np.ndarray:
    """
    Build a (N, 3+F) feature matrix from XYZ and selected extra dimensions.

    Parameters
    ----------
    xyz : (N, 3) float64
    dims : dict of dim_name -> (N,) array
    feature_names : list of extra feature names to append after xyz
    normalize_xyz : if True, zero-centre the XY coordinates and shift Z
                    so the minimum is 0 (relative height).

    Returns
    -------
    features : (N, 3+F) float32 array
    """
    n = len(xyz)
    coords = xyz.copy()

    if normalize_xyz:
        coords[:, :2] -= coords[:, :2].mean(axis=0)
        coords[:, 2] -= coords[:, 2].min()

    channels = [coords]

    for name in feature_names:
        if name in dims:
            arr = dims[name].astype(np.float32)
        elif name in NIR_ALIASES:
            arr = _resolve_nir(dims)
            if arr is None:
                logger.warning("NIR requested but not found; filling with zeros")
                arr = np.zeros(n, dtype=np.float32)
            else:
                arr = arr.astype(np.float32)
        else:
            logger.warning("Feature %r not found in LAS dims; filling with zeros", name)
            arr = np.zeros(n, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        channels.append(arr)

    features = np.concatenate(channels, axis=1).astype(np.float32)
    logger.debug("Feature matrix: shape=%s, names=[xyz, %s]", features.shape,
                 ", ".join(feature_names))
    return features


def feature_dim(cfg: dict) -> int:
    """Total feature dimensionality (3 for xyz + extras)."""
    return 3 + len(resolve_feature_list(cfg))
