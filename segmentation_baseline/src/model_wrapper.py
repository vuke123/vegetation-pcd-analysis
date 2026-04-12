"""
Model wrapper for RandLA-Net via Open3D-ML (PyTorch backend).

This module provides a clean interface to initialize, train, and run inference
with the RandLA-Net architecture from Open3D-ML. It wraps Open3D-ML's model
so it can be used with our custom TileDataset and config.

IMPORTANT: This is a baseline scaffold. The model is initialized with random
weights. A trained checkpoint is required for meaningful inference. Training
requires labeled point-cloud tiles.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _try_import_open3d_ml():
    """
    Import Open3D-ML with PyTorch backend.

    Returns the ml3d.torch module, or raises ImportError with guidance.
    """
    try:
        import open3d.ml.torch as ml3d  # noqa: F401
        return ml3d
    except ImportError as e:
        raise ImportError(
            "Open3D-ML with PyTorch backend is required.\n"
            "Install with:\n"
            "  pip install open3d open3d-ml\n"
            "Make sure PyTorch is installed first.\n"
            f"Original error: {e}"
        ) from e


def build_model(cfg: dict) -> nn.Module:
    """
    Build a RandLA-Net model from config.

    Parameters
    ----------
    cfg : dict — full config (model section is read from cfg['model'])

    Returns
    -------
    model : nn.Module
    """
    ml3d = _try_import_open3d_ml()

    model_cfg = cfg["model"]
    num_classes = cfg["num_classes"]

    # Open3D-ML's RandLANet accepts a config object.
    # We build it from our YAML config.
    randla_cfg = ml3d.models.RandLANet(
        name="RandLANet",
        num_classes=num_classes,
        num_points=model_cfg.get("num_points", 65536),
        num_layers=model_cfg.get("num_layers", 4),
        num_neighbors=model_cfg.get("num_neighbors", 16),
        sub_sampling_ratio=model_cfg.get("sub_sampling_ratio", [4, 4, 4, 4]),
        d_out=model_cfg.get("d_out", [16, 64, 128, 256]),
        in_channels=model_cfg.get("d_feature", 8),
    )

    logger.info("Built RandLA-Net: %d classes, %d layers, in_channels=%d",
                num_classes, model_cfg.get("num_layers", 4),
                model_cfg.get("d_feature", 8))

    return randla_cfg


def load_checkpoint(model: nn.Module, path: str | Path) -> nn.Module:
    """
    Load a saved checkpoint into the model.

    Raises FileNotFoundError with a clear message if the checkpoint does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "You need to train the model first with train_randlanet.py,\n"
            "or provide a pretrained checkpoint path in the config."
        )
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %s)",
                path, state.get("epoch", "?"))
    return model


def save_checkpoint(model: nn.Module,
                    optimizer,
                    epoch: int,
                    path: str | Path,
                    extra: dict | None = None) -> None:
    """Save model + optimizer state to a checkpoint file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    torch.save(state, str(path))
    logger.info("Saved checkpoint: %s (epoch %d)", path, epoch)
