#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

try:
    import open3d as o3d  # type: ignore
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("open3d is required. Install it with `pip install open3d`.") from exc

from pointcloud_transformer_autoencoder import (
    configure_gpu,
    build_dataset,
    build_point_transformer_autoencoder,
    load_xyz_from_pcd,
)


def normalize_with_params(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize points to unit sphere and return normalization params.

    Returns
    -------
    norm : (N, 3) float32
    centroid : (1, 3) float32
    scale : float
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got shape {points.shape}")

    centroid = points.mean(axis=0, keepdims=True).astype(np.float32)
    shifted = points.astype(np.float32) - centroid
    dists = np.linalg.norm(shifted, axis=1)
    max_dist = float(dists.max()) if shifted.shape[0] > 0 else 1.0
    if max_dist > 0.0:
        norm = shifted / max_dist
    else:
        norm = shifted
    return norm.astype(np.float32), centroid, max_dist


def choose_target_pcd(cluster_dir: str, target_pcd: str) -> str:
    """Resolve which PCD file to use for the reconstruction demo."""
    cluster_dir_path = Path(cluster_dir)
    if target_pcd:
        p = Path(target_pcd)
        if not p.is_file():
            # try relative to cluster_dir
            p = cluster_dir_path / target_pcd
        if not p.is_file():
            raise FileNotFoundError(f"Target PCD not found: {target_pcd}")
        return str(p)

    # Fallback: try some typical patterns from your pipeline
    patterns = [
        "config1_leaf00cm_tol40cm_cluster_00.pcd",
        "config*_cluster_00.pcd",
        "config*_cluster_*.pcd",
        "*.pcd",
    ]
    for pat in patterns:
        matches = sorted(cluster_dir_path.glob(pat))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(f"No PCD files found in {cluster_dir}.")


def reconstruct_from_downsample(
    model: tf.keras.Model,
    pcd_path: str,
    num_points: int,
    keep_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a point cloud and reconstruct it with the autoencoder.

    Parameters
    ----------
    model : tf.keras.Model
        Trained autoencoder.
    pcd_path : str
        Path to the original PCD.
    num_points : int
        Number of points expected by the model.
    keep_ratio : float
        Fraction of points to keep when downsampling (e.g. 0.3 for 70% drop).

    Returns
    -------
    original_full : (N, 3)
    downsampled_full : (K, 3)
    reconstructed_full : (num_points, 3)
    """
    coords_full = load_xyz_from_pcd(pcd_path).astype(np.float32)
    if coords_full.shape[0] == 0:
        raise ValueError(f"PCD {pcd_path} has no points.")

    norm_full, centroid, scale = normalize_with_params(coords_full)

    n = norm_full.shape[0]
    keep_n = max(1, int(n * keep_ratio))
    if keep_n >= n:
        keep_n = n
    idx_keep = np.random.choice(n, keep_n, replace=False)
    norm_down = norm_full[idx_keep]

    # Build model input by sampling num_points from the downsampled set
    if keep_n < num_points:
        idx_sample = np.random.choice(keep_n, num_points, replace=True)
    else:
        idx_sample = np.random.choice(keep_n, num_points, replace=False)
    input_norm = norm_down[idx_sample][None, ...]  # (1, num_points, 3)

    recon_norm = model.predict(input_norm, verbose=0)[0]  # (num_points, 3)

    # Denormalize back to original coordinate system
    reconstructed_full = recon_norm * scale + centroid  # broadcast centroid
    downsampled_full = norm_down * scale + centroid
    original_full = coords_full

    return original_full, downsampled_full, reconstructed_full


def make_o3d_pcd(points: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = np.tile(np.array(color, dtype=np.float64), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def save_pcd(path: str, points: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pcd = make_o3d_pcd(points, (0.5, 0.5, 0.5))
    o3d.io.write_point_cloud(str(p), pcd)


def visualize_triplet(
    original_full: np.ndarray,
    downsampled_full: np.ndarray,
    reconstructed_full: np.ndarray,
) -> None:
    """Show original, downsampled, and reconstructed clouds side by side."""
    import numpy as _np

    pcd_orig = make_o3d_pcd(original_full, (0.7, 0.7, 0.7))
    pcd_down = make_o3d_pcd(downsampled_full, (0.2, 0.4, 0.9))
    pcd_recon = make_o3d_pcd(reconstructed_full, (0.9, 0.3, 0.2))

    bbox = pcd_orig.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()[0]
    shift = _np.array([extent * 2.0, 0.0, 0.0])

    pcd_down.translate(shift)
    pcd_recon.translate(shift * 2.0)

    o3d.visualization.draw_geometries(
        [pcd_orig, pcd_down, pcd_recon],
        window_name="Original (gray), Downsampled (blue), Reconstructed (red)",
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train/load the transformer autoencoder on cluster PCDs, "
            "then downsample one PCD by 70% and reconstruct it."
        )
    )
    ap.add_argument(
        "--cluster-dir",
        type=str,
        default="./out_cluster",
        help=(
            "Directory containing cluster .pcd files. "
            "If you run this from the project root, use e.g. scripts/out_cluster."
        ),
    )
    ap.add_argument(
        "--target-pcd",
        type=str,
        default="",
        help=(
            "Specific PCD file to use for the demo. "
            "If empty, a suitable config*_cluster_*.pcd will be chosen automatically."
        ),
    )
    ap.add_argument(
        "--keep-ratio",
        type=float,
        default=0.3,
        help="Fraction of points to keep when downsampling (0.3 means drop 70%).",
    )
    ap.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of points per sample expected by the autoencoder.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    ap.add_argument(
        "--steps-per-epoch",
        type=int,
        default=200,
        help="Steps per epoch. Set <=0 to auto-compute from number of clusters.",
    )
    ap.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer feature dimension.",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer blocks in encoder/decoder.",
    )
    ap.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads.",
    )
    ap.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent feature dimension.",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate.",
    )
    ap.add_argument(
        "--model-path",
        type=str,
        default="./ae_checkpoints/demo_model.keras",
        help="Path to save/load the trained model.",
    )
    ap.add_argument(
        "--reuse-model",
        action="store_true",
        help=(
            "If set, skip training and load an existing model from --model-path "
            "(must exist)."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="./ae_recon_out",
        help="Directory to store downsampled and reconstructed PCD files.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.keep_ratio <= 1.0):
        raise ValueError("--keep-ratio must be in (0, 1].")

    configure_gpu()

    model_path = Path(args.model_path)

    if args.reuse_model and model_path.is_file():
        print(f"[INFO] Loading existing model from {model_path}")
        model = tf.keras.models.load_model(str(model_path))
    else:
        print(f"[INFO] Building dataset from cluster dir: {args.cluster_dir}")
        ds, num_files = build_dataset(
            cluster_dir=args.cluster_dir,
            num_points=args.num_points,
            batch_size=args.batch_size,
            augment=True,
        )
        print(f"[INFO] Found {num_files} cluster PCD files.")

        print(
            f"[INFO] Building model: num_points={args.num_points}, d_model={args.d_model}, "
            f"num_layers={args.num_layers}, num_heads={args.num_heads}, latent_dim={args.latent_dim}"
        )
        model = build_point_transformer_autoencoder(
            num_points=args.num_points,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            latent_dim=args.latent_dim,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss="mse",
            metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
        )

        if args.steps_per_epoch <= 0:
            steps_per_epoch = max(1, num_files // max(1, args.batch_size))
        else:
            steps_per_epoch = args.steps_per_epoch

        print(
            f"[INFO] Starting training: epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, "
            f"batch_size={args.batch_size}"
        )
        model.fit(
            ds,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
        )

        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
        print(f"[INFO] Saved trained model to {model_path}")

    target_pcd_path = choose_target_pcd(args.cluster_dir, args.target_pcd)
    print(f"[INFO] Using target PCD: {target_pcd_path}")

    original_full, downsampled_full, reconstructed_full = reconstruct_from_downsample(
        model,
        target_pcd_path,
        num_points=args.num_points,
        keep_ratio=args.keep_ratio,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(target_pcd_path).stem

    down_pcd_path = out_dir / f"{base}_downsampled_{int(args.keep_ratio * 100)}.pcd"
    recon_pcd_path = out_dir / f"{base}_reconstructed.pcd"

    save_pcd(str(down_pcd_path), downsampled_full)
    save_pcd(str(recon_pcd_path), reconstructed_full)

    print(f"[INFO] Wrote downsampled PCD: {down_pcd_path}")
    print(f"[INFO] Wrote reconstructed PCD: {recon_pcd_path}")

    visualize_triplet(original_full, downsampled_full, reconstructed_full)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
