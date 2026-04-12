"""
PyTorch Dataset for tiled point-cloud semantic segmentation.

This dataset reads pre-tiled .npz files produced by tile_builder and
presents them to the DataLoader. It is designed to be compatible with
Open3D-ML's training pipeline but can also be used standalone with a
plain PyTorch training loop.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .tile_builder import load_tile

logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """
    Dataset of pre-tiled .npz point-cloud blocks.

    Each item returns:
        features : (N, F) float32 tensor
        labels   : (N,) int64 tensor
        meta     : dict with source_file, tile_origin, path
    """

    def __init__(self,
                 tile_paths: List[Path],
                 max_points: int = 65536,
                 augment: bool = False):
        """
        Parameters
        ----------
        tile_paths : list of paths to .npz tile files
        max_points : pad or subsample to exactly this many points
        augment : if True, apply random rotation + jitter (training only)
        """
        self.tile_paths = sorted(tile_paths)
        self.max_points = max_points
        self.augment = augment

        if not self.tile_paths:
            logger.warning("TileDataset created with 0 tile paths.")

    def __len__(self) -> int:
        return len(self.tile_paths)

    def __getitem__(self, idx: int) -> dict:
        tile = load_tile(self.tile_paths[idx])
        features = tile["features"].astype(np.float32)
        labels = tile["labels"].astype(np.int64)
        n = len(features)

        # Pad or subsample to fixed size.
        if n > self.max_points:
            choice = np.random.choice(n, self.max_points, replace=False)
            features = features[choice]
            labels = labels[choice]
        elif n < self.max_points:
            pad = self.max_points - n
            choice = np.random.choice(n, pad, replace=True)
            features = np.concatenate([features, features[choice]], axis=0)
            labels = np.concatenate([labels, labels[choice]], axis=0)

        if self.augment:
            features = self._augment(features)

        return {
            "features": torch.from_numpy(features),
            "labels": torch.from_numpy(labels),
            "path": str(self.tile_paths[idx]),
        }

    @staticmethod
    def _augment(features: np.ndarray) -> np.ndarray:
        """Random rotation around Z axis + small XYZ jitter."""
        theta = np.random.uniform(0, 2 * np.pi)
        cos, sin = np.cos(theta), np.sin(theta)
        features = features.copy()
        x = features[:, 0] * cos - features[:, 1] * sin
        y = features[:, 0] * sin + features[:, 1] * cos
        features[:, 0] = x
        features[:, 1] = y
        features[:, :3] += np.random.normal(0, 0.01, size=features[:, :3].shape).astype(np.float32)
        return features


def load_split(split_file: Path, tile_dir: Path) -> List[Path]:
    """
    Read a split file (one tile filename per line) and return resolved paths.

    Lines starting with '#' are ignored. Missing files are skipped with a warning.
    """
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. "
            "Create it with one tile filename per line."
        )
    paths = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = tile_dir / line
        if p.is_file():
            paths.append(p)
        else:
            logger.warning("Tile listed in split but not found: %s", p)
    return paths


def has_any_labels(tile_paths: List[Path], sample_n: int = 10) -> bool:
    """
    Quick check: do any of the sampled tiles contain non-zero labels?

    Used to detect whether labeled data is actually available before training.
    """
    check = tile_paths[:sample_n]
    for p in check:
        tile = load_tile(p)
        if np.any(tile["labels"] > 0):
            return True
    return False
