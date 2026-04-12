"""
Segmentation evaluation metrics.

Computes per-class IoU, mean IoU, overall accuracy, and confusion matrix
for semantic segmentation of point clouds.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def compute_confusion_matrix(y_true: np.ndarray,
                             y_pred: np.ndarray,
                             num_classes: int,
                             ignore_label: int = 0) -> np.ndarray:
    """
    Compute confusion matrix, optionally ignoring a label.

    Parameters
    ----------
    y_true, y_pred : (N,) int arrays
    num_classes : number of valid classes (excluding ignored)
    ignore_label : label id to exclude from evaluation

    Returns
    -------
    cm : (num_classes, num_classes) int array
         rows = true, cols = predicted. Class indices are 1-based labels
         mapped to 0-based matrix rows (label 1 -> row 0, etc.).
    """
    mask = y_true != ignore_label
    yt = y_true[mask] - 1   # shift to 0-based
    yp = y_pred[mask] - 1

    valid = (yt >= 0) & (yt < num_classes) & (yp >= 0) & (yp < num_classes)
    yt = yt[valid]
    yp = yp[valid]

    cm = sk_confusion_matrix(yt, yp, labels=list(range(num_classes)))
    return cm


def iou_per_class(cm: np.ndarray) -> np.ndarray:
    """
    Per-class Intersection-over-Union from a confusion matrix.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    iou = np.where(denom > 0, tp / denom, 0.0)
    return iou


def mean_iou(cm: np.ndarray) -> float:
    """Mean IoU across classes that have at least one ground-truth sample."""
    ious = iou_per_class(cm)
    present = cm.sum(axis=1) > 0
    if not np.any(present):
        return 0.0
    return float(np.mean(ious[present]))


def overall_accuracy(cm: np.ndarray) -> float:
    """Overall pixel/point accuracy."""
    total = cm.sum()
    if total == 0:
        return 0.0
    return float(np.diag(cm).sum() / total)


def classification_report(cm: np.ndarray,
                          class_names: List[str]) -> Dict:
    """
    Build a structured report from the confusion matrix.

    Returns dict with per-class IoU, mIoU, and OA.
    """
    ious = iou_per_class(cm)
    report = {
        "overall_accuracy": overall_accuracy(cm),
        "mean_iou": mean_iou(cm),
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        n_gt = int(cm[i].sum())
        report["per_class"][name] = {
            "iou": float(ious[i]),
            "n_ground_truth": n_gt,
        }
    return report


def print_report(report: Dict, class_names: List[str]) -> None:
    """Pretty-print a classification report."""
    print(f"\n{'='*50}")
    print(f"  Overall accuracy:  {report['overall_accuracy']:.4f}")
    print(f"  Mean IoU:          {report['mean_iou']:.4f}")
    print(f"{'='*50}")
    print(f"  {'Class':<15} {'IoU':>8}  {'GT points':>10}")
    print(f"  {'-'*35}")
    for name in class_names:
        info = report["per_class"].get(name, {})
        iou = info.get("iou", 0.0)
        n = info.get("n_ground_truth", 0)
        print(f"  {name:<15} {iou:>8.4f}  {n:>10,}")
    print()
