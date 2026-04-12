#!/usr/bin/env python3
"""
Run inference with a trained RandLA-Net checkpoint.

Predicts semantic labels for one LAS file or all LAS files in a directory.
Outputs labeled LAS files to the predictions directory.

IMPORTANT: A trained checkpoint is required. If no checkpoint exists, this
script will exit with a clear message instead of producing fake predictions.

Usage:
    python scripts/infer_randlanet.py --config config/randlanet_config.yaml --input data/raw/scene.las
    python scripts/infer_randlanet.py --config config/randlanet_config.yaml --input data/raw/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logging, resolve_path, ensure_dir
from src.io_las import read_las, write_las_with_labels
from src.features import resolve_feature_list, extract_features, feature_dim
from src.tile_builder import (compute_tile_origins, extract_tile,
                              build_tiles_for_scene, load_tile)
from src.model_wrapper import build_model, load_checkpoint
from src.label_mapping import LABEL_MAP

import logging
logger = logging.getLogger(__name__)


def infer_scene(model: torch.nn.Module,
                xyz: np.ndarray,
                features: np.ndarray,
                tile_cfg: dict,
                device: torch.device,
                max_points: int) -> np.ndarray:
    """
    Run inference on a single scene by tiling, predicting, and reassembling.

    Returns per-point label predictions for the full scene.
    """
    block_size = tile_cfg["block_size"]
    overlap = tile_cfg["overlap"]
    min_pts = tile_cfg["min_points_per_tile"]

    xy_min = xyz[:, :2].min(axis=0)
    xy_max = xyz[:, :2].max(axis=0)
    origins = compute_tile_origins(xy_min, xy_max, block_size, overlap)

    # For each point, track the best prediction (from the nearest tile centre).
    labels = np.zeros(len(xyz), dtype=np.int32)
    best_dist = np.full(len(xyz), np.inf)
    half = block_size / 2.0

    model.eval()
    with torch.no_grad():
        for origin in tqdm(origins, desc="  Tiles", leave=False):
            # Find points in this tile.
            x0, y0 = origin
            mask = (
                (xyz[:, 0] >= x0) & (xyz[:, 0] < x0 + block_size) &
                (xyz[:, 1] >= y0) & (xyz[:, 1] < y0 + block_size)
            )
            idx = np.where(mask)[0]
            if len(idx) < min_pts:
                continue

            # Subsample if needed.
            if len(idx) > max_points:
                sub = np.random.choice(len(idx), max_points, replace=False)
                tile_idx = idx[sub]
            else:
                tile_idx = idx

            # Pad to max_points.
            tile_feat = features[tile_idx]
            if len(tile_feat) < max_points:
                pad_n = max_points - len(tile_feat)
                pad_choice = np.random.choice(len(tile_feat), pad_n, replace=True)
                tile_feat = np.concatenate([tile_feat, tile_feat[pad_choice]], axis=0)

            # Predict.
            feat_tensor = torch.from_numpy(tile_feat).unsqueeze(0).to(device)
            logits = model(feat_tensor)
            preds = logits.argmax(dim=-1).cpu().numpy().flatten()

            # Keep only predictions for real (non-padded) points.
            preds = preds[:len(tile_idx)]

            # Assign using nearest-centre voting.
            centre = origin + half
            d = np.sqrt(
                (xyz[tile_idx, 0] - centre[0])**2 +
                (xyz[tile_idx, 1] - centre[1])**2
            )
            update = d < best_dist[tile_idx]
            update_idx = tile_idx[update]
            labels[update_idx] = preds[update]
            best_dist[update_idx] = d[update]

    return labels


def main():
    parser = argparse.ArgumentParser(description="Run RandLA-Net inference.")
    parser.add_argument("--config", default="config/randlanet_config.yaml")
    parser.add_argument("--input", required=True,
                        help="Input LAS file or directory of LAS files")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path from config")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(resolve_path(ROOT, args.config))

    # --- Checkpoint ----------------------------------------------------------
    ckpt_path = args.checkpoint or cfg["inference"]["checkpoint"]
    ckpt_path = resolve_path(ROOT, ckpt_path)

    if not ckpt_path.is_file():
        print("\n" + "="*60)
        print("  INFERENCE CANNOT RUN: checkpoint not found")
        print("="*60)
        print(f"\n  Expected checkpoint at: {ckpt_path}")
        print("\n  You need to train the model first:")
        print("    1. Prepare labeled tiles: python scripts/prepare_dataset.py")
        print("    2. Train: python scripts/train_randlanet.py")
        print("    3. Re-run this script")
        print("\n  No predictions will be produced without a trained model.")
        sys.exit(1)

    # --- Model ---------------------------------------------------------------
    cfg["model"]["d_feature"] = feature_dim(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg)
    model = load_checkpoint(model, ckpt_path)
    model = model.to(device)

    # --- Input files ---------------------------------------------------------
    input_path = Path(args.input)
    if input_path.is_dir():
        las_files = sorted(input_path.glob("*.la[sz]"))
    elif input_path.is_file():
        las_files = [input_path]
    else:
        print(f"ERROR: Input not found: {input_path}")
        sys.exit(1)

    if not las_files:
        print(f"No LAS/LAZ files found in {input_path}")
        sys.exit(1)

    pred_dir = resolve_path(ROOT, cfg["output"]["predictions_dir"])
    ensure_dir(pred_dir)

    feature_names = resolve_feature_list(cfg)
    tile_cfg = cfg["tiling"]
    max_points = cfg["model"].get("num_points", 65536)

    # --- Inference -----------------------------------------------------------
    print(f"\nRunning inference on {len(las_files)} file(s)")

    for las_path in las_files:
        print(f"\n  Processing: {las_path.name}")
        data = read_las(las_path)
        features = extract_features(data["xyz"], data["dims"], feature_names,
                                    normalize_xyz=True)

        labels = infer_scene(model, data["xyz"], features, tile_cfg,
                             device, max_points)

        # Write output.
        out_path = pred_dir / f"{las_path.stem}_pred.las"
        write_las_with_labels(data["xyz"], labels, out_path)

        # Summary.
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Predicted labels:")
        for u, c in zip(unique, counts):
            name = LABEL_MAP.get(int(u), f"unknown_{u}")
            print(f"    {u} ({name}): {c:,} points")

    print(f"\nPredictions saved to: {pred_dir}")


if __name__ == "__main__":
    main()
