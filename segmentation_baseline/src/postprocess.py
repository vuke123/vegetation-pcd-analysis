"""
Post-processing utilities for segmentation predictions.

Includes functions to reassemble tile-level predictions back into
full-scene point clouds and apply simple spatial smoothing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from .tile_builder import load_tile

logger = logging.getLogger(__name__)


def merge_tile_predictions(tile_paths: List[Path],
                           predictions: List[np.ndarray],
                           original_xyz: np.ndarray,
                           tile_origins: List[np.ndarray],
                           block_size: float,
                           overlap: float) -> np.ndarray:
    """
    Merge per-tile predictions back into a full-scene label array.

    For points that fall in the overlap region between tiles, the prediction
    from the tile whose centre is closest is used (nearest-centre voting).

    Parameters
    ----------
    tile_paths : list of tile .npz paths (for reading tile_origin metadata)
    predictions : list of (N_i,) int arrays, one per tile
    original_xyz : (N_total, 3) — the full scene's XYZ coordinates
    tile_origins : list of (2,) arrays — XY origins per tile
    block_size : tile size in metres
    overlap : overlap in metres

    Returns
    -------
    labels : (N_total,) int array — predicted label per point (0 = unassigned)
    """
    n = len(original_xyz)
    labels = np.zeros(n, dtype=np.int32)
    dist_to_centre = np.full(n, np.inf)

    half = block_size / 2.0

    for pred, origin in zip(predictions, tile_origins):
        centre = origin + half
        # Find points within this tile's footprint.
        dx = original_xyz[:, 0] - centre[0]
        dy = original_xyz[:, 1] - centre[1]
        in_tile = (
            (np.abs(dx) <= half) &
            (np.abs(dy) <= half)
        )
        d = np.sqrt(dx**2 + dy**2)

        # Assign prediction where this tile is closer to the point than
        # any previously assigned tile.
        update = in_tile & (d < dist_to_centre)
        idx = np.where(update)[0]

        # pred may have been padded/subsampled — we can only assign as many
        # as we have predictions. This is a simplified merge; a production
        # version would use spatial indexing.
        # TODO: implement proper spatial lookup for padded tiles.
        labels[idx[:len(pred)]] = pred[:len(idx)]
        dist_to_centre[idx[:len(pred)]] = d[idx[:len(pred)]]

    n_assigned = np.count_nonzero(labels)
    logger.info("Merged predictions: %d / %d points assigned (%.1f%%)",
                n_assigned, n, 100 * n_assigned / max(n, 1))
    return labels


def majority_vote_smooth(labels: np.ndarray,
                         xyz: np.ndarray,
                         radius: float = 0.2,
                         min_neighbors: int = 3) -> np.ndarray:
    """
    Simple spatial majority-vote smoothing.

    For each point, find neighbours within `radius` and replace the label
    with the majority vote if enough neighbours agree.

    NOTE: This is a naive O(N*k) implementation. For large clouds, use a
    KD-tree from scipy or open3d. Placeholder for future optimization.
    """
    # TODO: implement with scipy.spatial.KDTree for production use.
    # For now, return labels unchanged — this is a scaffold.
    logger.info("majority_vote_smooth: not yet implemented, returning labels unchanged")
    return labels.copy()
