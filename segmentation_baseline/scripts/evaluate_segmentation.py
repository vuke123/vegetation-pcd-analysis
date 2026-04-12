#!/usr/bin/env python3
"""
Evaluate segmentation predictions against ground-truth labels.

Computes per-class IoU, mean IoU, overall accuracy, and confusion matrix.
Works on either:
  - Tile-level: predicted vs. ground-truth tiles
  - Scene-level: predicted LAS vs. ground-truth LAS

IMPORTANT: Evaluation requires ground-truth labels. If no labels are found,
this script will exit with a clear message.

Usage:
    python scripts/evaluate_segmentation.py --config config/randlanet_config.yaml
    python scripts/evaluate_segmentation.py --pred-dir output/predictions --gt-dir data/raw
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logging, resolve_path
from src.io_las import read_las, extract_labels
from src.label_mapping import class_names, NUM_CLASSES
from src.metrics import (compute_confusion_matrix, classification_report,
                         print_report)

import logging
logger = logging.getLogger(__name__)


def evaluate_scene_pair(pred_path: Path, gt_path: Path) -> np.ndarray | None:
    """
    Evaluate one predicted LAS against its ground-truth LAS.

    Returns a confusion matrix, or None if no GT labels exist.
    """
    pred_data = read_las(pred_path)
    gt_data = read_las(gt_path)

    pred_labels = extract_labels(pred_data["dims"])
    gt_labels = extract_labels(gt_data["dims"])

    if gt_labels is None or not np.any(gt_labels > 0):
        logger.warning("No GT labels in %s — skipping", gt_path.name)
        return None

    if pred_labels is None:
        logger.warning("No predictions in %s — skipping", pred_path.name)
        return None

    if len(pred_labels) != len(gt_labels):
        logger.warning("Point count mismatch: pred=%d, gt=%d for %s",
                        len(pred_labels), len(gt_labels), gt_path.name)
        # Use the shorter length.
        n = min(len(pred_labels), len(gt_labels))
        pred_labels = pred_labels[:n]
        gt_labels = gt_labels[:n]

    cm = compute_confusion_matrix(gt_labels, pred_labels, NUM_CLASSES)
    return cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate semantic segmentation.")
    parser.add_argument("--config", default="config/randlanet_config.yaml")
    parser.add_argument("--pred-dir", default=None,
                        help="Directory of predicted LAS files (override config)")
    parser.add_argument("--gt-dir", default=None,
                        help="Directory of ground-truth LAS files (override config)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(resolve_path(ROOT, args.config))

    pred_dir = Path(args.pred_dir) if args.pred_dir else resolve_path(ROOT, cfg["output"]["predictions_dir"])
    gt_dir = Path(args.gt_dir) if args.gt_dir else resolve_path(ROOT, cfg["data"]["raw_dir"])

    if not pred_dir.is_dir():
        print(f"\nPrediction directory not found: {pred_dir}")
        print("Run infer_randlanet.py first to generate predictions.")
        sys.exit(1)

    pred_files = sorted(pred_dir.glob("*_pred.la[sz]"))
    if not pred_files:
        print(f"\nNo prediction files (*_pred.las) found in {pred_dir}")
        print("Run infer_randlanet.py first.")
        sys.exit(1)

    print(f"\nEvaluating {len(pred_files)} prediction file(s)")
    print(f"  Predictions: {pred_dir}")
    print(f"  Ground truth: {gt_dir}")

    # Accumulate confusion matrix across all scenes.
    total_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    n_evaluated = 0

    for pred_path in pred_files:
        # Match prediction to ground truth by stem.
        # Prediction: scene_001_pred.las -> GT: scene_001.las
        gt_stem = pred_path.stem.replace("_pred", "")
        gt_candidates = list(gt_dir.glob(f"{gt_stem}.la[sz]"))

        if not gt_candidates:
            logger.warning("No GT file found for %s", pred_path.name)
            continue

        gt_path = gt_candidates[0]
        print(f"\n  {pred_path.name}  vs  {gt_path.name}")

        cm = evaluate_scene_pair(pred_path, gt_path)
        if cm is not None:
            total_cm += cm
            n_evaluated += 1

            # Per-scene report.
            report = classification_report(cm, class_names())
            print(f"    OA={report['overall_accuracy']:.4f}, mIoU={report['mean_iou']:.4f}")

    if n_evaluated == 0:
        print("\n" + "="*60)
        print("  EVALUATION FAILED: no labeled ground-truth data found")
        print("="*60)
        print("\n  The prediction files exist, but no matching ground-truth")
        print("  LAS files with semantic labels were found.")
        print("\n  To evaluate, you need:")
        print("    1. Ground-truth LAS files with a 'label' or 'classification' dimension")
        print("    2. File names matching: <scene>_pred.las -> <scene>.las")
        sys.exit(1)

    # --- Global report -------------------------------------------------------
    print("\n" + "="*60)
    print(f"  GLOBAL RESULTS ({n_evaluated} scene(s))")
    print("="*60)

    names = class_names()
    report = classification_report(total_cm, names)
    print_report(report, names)

    # Print raw confusion matrix.
    print("Confusion matrix (rows=GT, cols=pred):")
    header = "         " + "  ".join(f"{n[:8]:>8}" for n in names)
    print(header)
    for i, name in enumerate(names):
        row = "  ".join(f"{total_cm[i, j]:>8d}" for j in range(NUM_CLASSES))
        print(f"  {name[:8]:<8} {row}")


if __name__ == "__main__":
    main()
