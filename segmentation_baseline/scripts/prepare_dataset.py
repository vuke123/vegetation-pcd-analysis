#!/usr/bin/env python3
"""
Dataset preparation: scan LAS files, validate features, build tiles.

This script is safe to run without labels — it will create unlabeled tiles
that are structurally ready for future training once labels are added.

Usage:
    python scripts/prepare_dataset.py --config config/randlanet_config.yaml
    python scripts/prepare_dataset.py --config config/randlanet_config.yaml --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from the segmentation_baseline/ directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logging, resolve_path, ensure_dir
from src.io_las import read_las, list_available_features, extract_labels
from src.features import resolve_feature_list, extract_features
from src.tile_builder import build_tiles_for_scene
from src.label_mapping import LABEL_MAP

import logging
logger = logging.getLogger(__name__)


def summarize_scene(data: dict, labels: np.ndarray | None, feature_names: list) -> None:
    """Print a summary of one scene's contents."""
    xyz = data["xyz"]
    dims = data["dims"]
    n = data["n_points"]

    print(f"\n  File:   {data['path']}")
    print(f"  Points: {n:,}")
    print(f"  XYZ range: X=[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]  "
          f"Y=[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]  "
          f"Z=[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    print(f"  Available dims: {sorted(dims.keys())}")

    # Check which requested features are present.
    present = [f for f in feature_names if f in dims]
    missing = [f for f in feature_names if f not in dims]
    print(f"  Requested features: {feature_names}")
    print(f"    present: {present}")
    if missing:
        print(f"    MISSING (will be zero-filled): {missing}")

    if labels is not None and np.any(labels > 0):
        unique, counts = np.unique(labels, return_counts=True)
        print("  Labels found:")
        for u, c in zip(unique, counts):
            name = LABEL_MAP.get(int(u), f"unknown_{u}")
            print(f"    {u} ({name}): {c:,} points")
    else:
        print("  Labels: NONE (all points will be marked unlabeled)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare tiled dataset from LAS/LAZ point clouds."
    )
    parser.add_argument("--config", default="config/randlanet_config.yaml",
                        help="Path to config YAML")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(resolve_path(ROOT, args.config))

    raw_dir = resolve_path(ROOT, cfg["data"]["raw_dir"])
    tile_dir = resolve_path(ROOT, cfg["data"]["tile_dir"])
    ensure_dir(tile_dir)

    feature_names = resolve_feature_list(cfg)
    tile_cfg = cfg["tiling"]

    # Scan for LAS/LAZ files.
    las_files = sorted(raw_dir.glob("*.la[sz]"))
    if not las_files:
        print(f"\nNo LAS/LAZ files found in {raw_dir}")
        print("Place your input point clouds there and re-run.")
        print(f"Expected structure: {raw_dir}/scene_001.las")
        sys.exit(0)

    print(f"\nFound {len(las_files)} point cloud file(s) in {raw_dir}")
    print(f"Feature mode: {cfg.get('feature_mode', 'xyz')} -> {['xyz'] + feature_names}")
    print(f"Tile config: block={tile_cfg['block_size']}m, overlap={tile_cfg['overlap']}m, "
          f"max_pts={tile_cfg['max_points_per_tile']}")

    total_tiles = 0

    for las_path in las_files:
        logger.info("Processing %s", las_path.name)
        data = read_las(las_path)
        labels = extract_labels(data["dims"])

        summarize_scene(data, labels, feature_names)

        # Build feature matrix.
        features = extract_features(
            data["xyz"], data["dims"], feature_names, normalize_xyz=True
        )

        # Tile the scene.
        n_tiles = build_tiles_for_scene(
            xyz=data["xyz"],
            features=features,
            labels=labels,
            source_name=las_path.name,
            tile_dir=tile_dir,
            block_size=tile_cfg["block_size"],
            overlap=tile_cfg["overlap"],
            max_points=tile_cfg["max_points_per_tile"],
            min_points=tile_cfg["min_points_per_tile"],
        )
        total_tiles += n_tiles

    print(f"\n{'='*50}")
    print(f"  Total tiles written: {total_tiles}")
    print(f"  Tile directory: {tile_dir}")
    print(f"{'='*50}")

    if total_tiles > 0:
        print("\nNext steps:")
        print("  1. Label your point clouds (e.g. in CloudCompare)")
        print("  2. Re-run this script to generate labeled tiles")
        print("  3. Create train/val/test splits in data/splits/")
        print("  4. Run train_randlanet.py")


if __name__ == "__main__":
    main()
