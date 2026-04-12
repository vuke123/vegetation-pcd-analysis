"""
Tile / block extraction from large point clouds.

Large scenes (e.g. a full vineyard at 100m+ extent) cannot be fed to
RandLA-Net as a single input. This module splits them into overlapping
rectangular tiles in the XY plane, with optional voxel subsampling and
point-count limits.

Each tile is saved as a .npz containing:
  points  : (N, 3) float64  — local XYZ (zero-centred)
  features: (N, F) float32  — per-point features
  labels  : (N,) int32      — semantic labels (0 = unlabeled)
  meta    : dict             — source file, tile origin, etc.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_tile_origins(xy_min: np.ndarray,
                         xy_max: np.ndarray,
                         block_size: float,
                         overlap: float) -> list[np.ndarray]:
    """
    Compute a regular grid of tile origins that covers the XY extent.

    Each origin is the bottom-left corner of a block_size x block_size tile.
    Adjacent tiles overlap by `overlap` metres.
    """
    step = block_size - overlap
    if step <= 0:
        raise ValueError(f"overlap ({overlap}) must be < block_size ({block_size})")

    origins = []
    x = xy_min[0]
    while x < xy_max[0]:
        y = xy_min[1]
        while y < xy_max[1]:
            origins.append(np.array([x, y]))
            y += step
        x += step

    return origins


def extract_tile(xyz: np.ndarray,
                 features: np.ndarray,
                 labels: np.ndarray | None,
                 origin: np.ndarray,
                 block_size: float,
                 max_points: int,
                 min_points: int) -> dict | None:
    """
    Extract one tile from a point cloud.

    Returns None if the tile has fewer than min_points after extraction.
    If the tile has more than max_points, randomly subsample.
    """
    x, y = origin
    mask = (
        (xyz[:, 0] >= x) & (xyz[:, 0] < x + block_size) &
        (xyz[:, 1] >= y) & (xyz[:, 1] < y + block_size)
    )
    idx = np.where(mask)[0]

    if len(idx) < min_points:
        return None

    # Random subsample if too many points.
    if len(idx) > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)

    tile_xyz = xyz[idx].copy()
    tile_feat = features[idx].copy()
    tile_labels = labels[idx].copy() if labels is not None else np.zeros(len(idx), dtype=np.int32)

    # Zero-centre the tile.
    tile_xyz[:, :2] -= origin
    # Features already have normalized xyz from extract_features, but
    # tile_xyz is the original-CRS subset. The feature xyz was already
    # normalized at scene level; here we just pass it through.

    return {
        "points": tile_xyz,
        "features": tile_feat,
        "labels": tile_labels,
        "n_points": len(idx),
    }


def save_tile(tile: dict,
              out_path: Path,
              source_file: str,
              origin: np.ndarray) -> None:
    """Save a tile as a compressed .npz file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        points=tile["points"],
        features=tile["features"],
        labels=tile["labels"],
        source_file=np.array(source_file),
        tile_origin=origin,
    )


def load_tile(path: Path) -> dict:
    """Load a tile .npz and return its contents as a dict."""
    data = np.load(str(path), allow_pickle=True)
    return {
        "points": data["points"],
        "features": data["features"],
        "labels": data["labels"],
        "source_file": str(data["source_file"]),
        "tile_origin": data["tile_origin"],
    }


def build_tiles_for_scene(xyz: np.ndarray,
                          features: np.ndarray,
                          labels: np.ndarray | None,
                          source_name: str,
                          tile_dir: Path,
                          block_size: float,
                          overlap: float,
                          max_points: int,
                          min_points: int) -> int:
    """
    Tile one scene and save all tiles to disk.

    Returns the number of tiles written.
    """
    xy_min = xyz[:, :2].min(axis=0)
    xy_max = xyz[:, :2].max(axis=0)
    origins = compute_tile_origins(xy_min, xy_max, block_size, overlap)

    stem = Path(source_name).stem
    count = 0

    for i, origin in enumerate(origins):
        tile = extract_tile(xyz, features, labels, origin,
                            block_size, max_points, min_points)
        if tile is None:
            continue

        out_path = tile_dir / f"{stem}_tile_{count:04d}.npz"
        save_tile(tile, out_path, source_name, origin)
        count += 1

    logger.info("Scene %s: %d tiles from %d grid positions", stem, count, len(origins))
    return count
