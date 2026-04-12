#!/usr/bin/env python3
"""
Train RandLA-Net on tiled, labeled point-cloud data.

This script loads the prepared tile dataset, initializes RandLA-Net via
Open3D-ML, and runs a training loop with validation.

IMPORTANT: Training requires labeled data. If no labeled tiles exist,
this script will exit with a clear message explaining what is needed.

Usage:
    python scripts/train_randlanet.py --config config/randlanet_config.yaml
    python scripts/train_randlanet.py --config config/randlanet_config.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, setup_logging, resolve_path, ensure_dir
from src.dataset import TileDataset, load_split, has_any_labels
from src.model_wrapper import build_model, save_checkpoint
from src.features import feature_dim
from src.label_mapping import class_names, NUM_CLASSES
from src.metrics import (compute_confusion_matrix, classification_report,
                         print_report, mean_iou)

import logging
logger = logging.getLogger(__name__)


def compute_class_weights(train_dataset: TileDataset,
                          num_classes: int,
                          sample_n: int = 50) -> torch.Tensor:
    """
    Estimate inverse-frequency class weights from a sample of training tiles.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    n = min(sample_n, len(train_dataset))
    for i in range(n):
        item = train_dataset[i]
        labs = item["labels"].numpy()
        for c in range(num_classes):
            counts[c] += np.sum(labs == (c + 1))  # labels are 1-based

    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    weights *= num_classes
    return torch.tensor(weights, dtype=torch.float32)


def validate(model: nn.Module,
             val_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> dict:
    """Run one validation pass and return loss + metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(features)
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=-1).cpu().numpy().flatten()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy().flatten())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    cm = compute_confusion_matrix(y_true, y_pred, NUM_CLASSES)

    return {
        "loss": total_loss / max(n_batches, 1),
        "miou": mean_iou(cm),
        "cm": cm,
    }


def main():
    parser = argparse.ArgumentParser(description="Train RandLA-Net for semantic segmentation.")
    parser.add_argument("--config", default="config/randlanet_config.yaml")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs from config")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(resolve_path(ROOT, args.config))

    tile_dir = resolve_path(ROOT, cfg["data"]["tile_dir"])
    split_dir = resolve_path(ROOT, cfg["data"]["split_dir"])
    ckpt_dir = resolve_path(ROOT, cfg["output"]["checkpoint_dir"])
    ensure_dir(ckpt_dir)

    # --- Load splits ---------------------------------------------------------
    train_split = split_dir / "train.txt"
    val_split = split_dir / "val.txt"

    if not train_split.is_file() or not val_split.is_file():
        print("\n" + "="*60)
        print("  TRAINING CANNOT START: split files missing")
        print("="*60)
        print(f"\n  Expected:")
        print(f"    {train_split}")
        print(f"    {val_split}")
        print(f"\n  Each file should contain one tile filename per line.")
        print(f"  Tile directory: {tile_dir}")
        print(f"\n  Steps to create splits:")
        print(f"    1. Run prepare_dataset.py to generate tiles")
        print(f"    2. Label your data")
        print(f"    3. Create train.txt and val.txt listing tile filenames")
        sys.exit(1)

    train_paths = load_split(train_split, tile_dir)
    val_paths = load_split(val_split, tile_dir)

    if not train_paths:
        print("ERROR: No valid training tiles found.")
        sys.exit(1)

    # --- Check for labels ----------------------------------------------------
    if not has_any_labels(train_paths):
        print("\n" + "="*60)
        print("  TRAINING CANNOT START: no labels found in training tiles")
        print("="*60)
        print("\n  The training tiles exist but contain only label=0 (unlabeled).")
        print("  Training a segmentation model requires per-point semantic labels.")
        print("\n  Next steps:")
        print("    1. Label your point clouds in CloudCompare or similar tool")
        print("    2. Re-run prepare_dataset.py to regenerate labeled tiles")
        print("    3. Re-run this script")
        sys.exit(1)

    print(f"\nTraining: {len(train_paths)} tiles")
    print(f"Validation: {len(val_paths)} tiles")

    # --- Dataset + DataLoader ------------------------------------------------
    train_cfg = cfg["training"]
    max_points = cfg["model"].get("num_points", 65536)

    train_ds = TileDataset(train_paths, max_points=max_points, augment=True)
    val_ds = TileDataset(val_paths, max_points=max_points, augment=False)

    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=2)

    # --- Model ---------------------------------------------------------------
    # Update feature dimension in config from the actual feature set.
    cfg["model"]["d_feature"] = feature_dim(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_model(cfg)
    model = model.to(device)

    # --- Optimizer + loss ----------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    # Class weights.
    if train_cfg.get("class_weights") == "auto":
        weights = compute_class_weights(train_ds, NUM_CLASSES)
        logger.info("Auto class weights: %s", weights.tolist())
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        ignore_index=0,  # ignore unlabeled
    )

    # --- Training loop -------------------------------------------------------
    epochs = args.epochs or train_cfg["epochs"]
    val_interval = train_cfg.get("val_interval", 5)
    patience = train_cfg.get("early_stopping_patience", 15)
    best_miou = 0.0
    patience_counter = 0

    print(f"\nStarting training: {epochs} epochs, val every {val_interval}")
    print(f"Classes: {class_names()}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits.view(-1, logits.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info("Epoch %d/%d — train loss: %.4f", epoch, epochs, avg_loss)

        # --- Validation ------------------------------------------------------
        if epoch % val_interval == 0 or epoch == epochs:
            val_result = validate(model, val_loader, criterion, device)
            print(f"  Epoch {epoch}: val_loss={val_result['loss']:.4f}, "
                  f"mIoU={val_result['miou']:.4f}")

            if val_result["miou"] > best_miou:
                best_miou = val_result["miou"]
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch,
                                ckpt_dir / "best_model.pth",
                                extra={"miou": best_miou})
                print(f"  -> New best mIoU: {best_miou:.4f}")
            else:
                patience_counter += val_interval
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch} (patience={patience})")
                    break

        # Save periodic checkpoint.
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch,
                            ckpt_dir / f"epoch_{epoch:03d}.pth")

    # --- Final report --------------------------------------------------------
    print(f"\nTraining complete. Best mIoU: {best_miou:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"\nNext: run infer_randlanet.py or evaluate_segmentation.py")


if __name__ == "__main__":
    main()
