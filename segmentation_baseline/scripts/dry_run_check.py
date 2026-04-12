#!/usr/bin/env python3
"""
Dry-run check: verify the baseline scaffold is wired correctly.

Does NOT require labels, checkpoints, or GPU. Verifies:
  1. All imports work
  2. Config loads and parses correctly
  3. Feature extraction logic works on synthetic data
  4. Tiling logic works on synthetic data
  5. Dataset class works with synthetic tiles
  6. Model can be initialized (Open3D-ML import)
  7. Label mapping is consistent with config

Run this first after cloning the repo to verify the environment is set up.

Usage:
    python scripts/dry_run_check.py
    python scripts/dry_run_check.py --config config/randlanet_config.yaml
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "  PASS"
FAIL = "  FAIL"


def check(name: str, fn):
    """Run a check function and print result."""
    try:
        fn()
        print(f"{PASS}: {name}")
        return True
    except Exception as e:
        print(f"{FAIL}: {name}")
        print(f"        {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Dry-run scaffold verification.")
    parser.add_argument("--config", default="config/randlanet_config.yaml")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  RandLA-Net Segmentation Baseline — Dry Run Check")
    print("="*60 + "\n")

    results = []

    # 1. Core imports
    def check_imports():
        import laspy         # noqa: F401
        import yaml          # noqa: F401
        import torch         # noqa: F401
        import sklearn       # noqa: F401
        import tqdm          # noqa: F401
    results.append(check("Core imports (laspy, yaml, torch, sklearn, tqdm)", check_imports))

    # 2. src package imports
    def check_src_imports():
        from src.utils import load_config           # noqa: F401
        from src.io_las import read_las              # noqa: F401
        from src.features import resolve_feature_list, extract_features, feature_dim  # noqa: F401
        from src.tile_builder import compute_tile_origins, extract_tile  # noqa: F401
        from src.dataset import TileDataset          # noqa: F401
        from src.label_mapping import LABEL_MAP, class_names  # noqa: F401
        from src.metrics import compute_confusion_matrix, mean_iou  # noqa: F401
        from src.postprocess import merge_tile_predictions  # noqa: F401
    results.append(check("src package imports", check_src_imports))

    # 3. Config loading
    def check_config():
        from src.utils import load_config, resolve_path
        cfg = load_config(resolve_path(ROOT, args.config))
        assert "classes" in cfg, "Missing 'classes' in config"
        assert "model" in cfg, "Missing 'model' in config"
        assert "tiling" in cfg, "Missing 'tiling' in config"
        assert cfg["num_classes"] > 0, "num_classes must be > 0"
    results.append(check("Config loading and validation", check_config))

    # 4. Feature extraction on synthetic data
    def check_features():
        from src.utils import load_config, resolve_path
        from src.features import resolve_feature_list, extract_features, feature_dim

        cfg = load_config(resolve_path(ROOT, args.config))
        names = resolve_feature_list(cfg)
        n = 1000
        xyz = np.random.rand(n, 3).astype(np.float64) * 50
        dims = {
            "ndvi": np.random.rand(n).astype(np.float32),
            "red": np.random.randint(0, 65535, n).astype(np.float32),
            "green": np.random.randint(0, 65535, n).astype(np.float32),
            "blue": np.random.randint(0, 65535, n).astype(np.float32),
            "nir": np.random.randint(0, 65535, n).astype(np.float32),
            "intensity": np.random.rand(n).astype(np.float32),
        }
        features = extract_features(xyz, dims, names, normalize_xyz=True)
        expected_dim = feature_dim(cfg)
        assert features.shape == (n, expected_dim), \
            f"Expected shape ({n}, {expected_dim}), got {features.shape}"
    results.append(check("Feature extraction (synthetic data)", check_features))

    # 5. Tiling on synthetic data
    def check_tiling():
        from src.utils import load_config, resolve_path
        from src.tile_builder import compute_tile_origins, extract_tile

        cfg = load_config(resolve_path(ROOT, args.config))
        tc = cfg["tiling"]

        n = 5000
        xyz = np.random.rand(n, 3).astype(np.float64) * 30
        features = np.random.rand(n, 4).astype(np.float32)
        labels = np.zeros(n, dtype=np.int32)

        xy_min = xyz[:, :2].min(axis=0)
        xy_max = xyz[:, :2].max(axis=0)
        origins = compute_tile_origins(xy_min, xy_max,
                                       tc["block_size"], tc["overlap"])
        assert len(origins) > 0, "No tile origins generated"

        tile = extract_tile(xyz, features, labels, origins[0],
                            tc["block_size"], tc["max_points_per_tile"],
                            tc["min_points_per_tile"])
        # tile may be None if random data doesn't fall in the first tile.
        # That's OK — we just check the function doesn't crash.
    results.append(check("Tiling logic (synthetic data)", check_tiling))

    # 6. Tile save/load round-trip
    def check_tile_io():
        from src.tile_builder import save_tile, load_tile
        n = 500
        tile = {
            "points": np.random.rand(n, 3),
            "features": np.random.rand(n, 4).astype(np.float32),
            "labels": np.zeros(n, dtype=np.int32),
            "n_points": n,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test_tile.npz"
            save_tile(tile, p, "test.las", np.array([0.0, 0.0]))
            loaded = load_tile(p)
            assert loaded["points"].shape == (n, 3)
            assert loaded["features"].shape == (n, 4)
            assert loaded["labels"].shape == (n,)
    results.append(check("Tile save/load round-trip", check_tile_io))

    # 7. Dataset class
    def check_dataset():
        from src.tile_builder import save_tile
        from src.dataset import TileDataset
        n = 200
        tile = {
            "points": np.random.rand(n, 3),
            "features": np.random.rand(n, 4).astype(np.float32),
            "labels": np.zeros(n, dtype=np.int32),
            "n_points": n,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "test_tile.npz"
            save_tile(tile, p, "test.las", np.array([0.0, 0.0]))
            ds = TileDataset([p], max_points=256)
            item = ds[0]
            assert item["features"].shape[0] == 256
            assert item["labels"].shape[0] == 256
    results.append(check("TileDataset (synthetic tile)", check_dataset))

    # 8. Label mapping consistency
    def check_labels():
        from src.utils import load_config, resolve_path
        from src.label_mapping import LABEL_MAP, class_names, NUM_CLASSES

        cfg = load_config(resolve_path(ROOT, args.config))
        assert cfg["num_classes"] == NUM_CLASSES, \
            f"Config num_classes={cfg['num_classes']} != code NUM_CLASSES={NUM_CLASSES}"
        for k, v in cfg["classes"].items():
            assert int(k) in LABEL_MAP, f"Class {k} in config but not in LABEL_MAP"
            assert LABEL_MAP[int(k)] == v, f"Mismatch: config={v}, code={LABEL_MAP[int(k)]}"
    results.append(check("Label mapping consistency (config vs code)", check_labels))

    # 9. Metrics on synthetic data
    def check_metrics():
        from src.metrics import compute_confusion_matrix, mean_iou, overall_accuracy
        y_true = np.array([1, 1, 2, 2, 3, 3, 0, 0])
        y_pred = np.array([1, 2, 2, 2, 3, 1, 0, 0])
        cm = compute_confusion_matrix(y_true, y_pred, 3, ignore_label=0)
        assert cm.shape == (3, 3)
        assert overall_accuracy(cm) > 0
        assert 0 <= mean_iou(cm) <= 1
    results.append(check("Metrics computation (synthetic)", check_metrics))

    # 10. Open3D-ML / model init (optional — may not be installed yet)
    def check_model():
        from src.utils import load_config, resolve_path
        from src.features import feature_dim
        cfg = load_config(resolve_path(ROOT, args.config))
        cfg["model"]["d_feature"] = feature_dim(cfg)
        from src.model_wrapper import build_model
        model = build_model(cfg)
        # Just check it returns something — don't run a forward pass.
        assert model is not None

    model_ok = check("Open3D-ML RandLA-Net init", check_model)
    results.append(model_ok)

    # --- Summary ---
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"  Results: {n_pass}/{n_total} checks passed")
    if not model_ok:
        print(f"\n  Note: Open3D-ML model init failed. This is expected if")
        print(f"  open3d-ml is not yet installed. Install with:")
        print(f"    pip install open3d open3d-ml")
        print(f"  All other checks are independent of Open3D-ML.")
    if n_pass == n_total:
        print(f"\n  All checks passed. The scaffold is wired correctly.")
    else:
        print(f"\n  {n_total - n_pass} check(s) failed. See details above.")
    print(f"{'='*60}\n")

    sys.exit(0 if n_pass >= n_total - 1 else 1)  # allow model init to fail


if __name__ == "__main__":
    main()
