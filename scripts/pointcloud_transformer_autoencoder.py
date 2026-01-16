#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import math
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "TensorFlow 2.x is required. Install it with `pip install tensorflow` or `pip install tensorflow-gpu`."
    ) from exc

try:
    from pypcd4 import PointCloud  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError(
        "pypcd4 is required to read PCD files. Install it with `pip install pypcd4`."
    ) from exc


def _pick_name(candidates: Iterable[str], available: Iterable[str]) -> str:
    available_set = set(available)
    for c in candidates:
        if c in available_set:
            return c
    raise KeyError(f"None of {list(candidates)} present in fields {list(available)}")


def load_xyz_from_pcd(path: str) -> np.ndarray:
    """Load XYZ coordinates from a PCD file using pypcd4.

    Supports both structured arrays (pc.pc_data) and plain numpy arrays.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"PCD not found: {p}")

    pc = PointCloud.from_path(str(p))

    fields = list(pc.fields)
    pc_data = getattr(pc, "pc_data", None)

    if pc_data is not None and getattr(pc_data.dtype, "names", None):
        names = list(pc_data.dtype.names)
        x_name = _pick_name(["x", "X"], names)
        y_name = _pick_name(["y", "Y"], names)
        z_name = _pick_name(["z", "Z"], names)
        coords = np.stack(
            [pc_data[x_name], pc_data[y_name], pc_data[z_name]], axis=-1
        ).astype(np.float32)
    else:
        x_name = _pick_name(["x", "X"], fields)
        y_name = _pick_name(["y", "Y"], fields)
        z_name = _pick_name(["z", "Z"], fields)
        arr = pc.numpy([x_name, y_name, z_name])
        coords = arr.astype(np.float32)

    return coords


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center points and scale to unit sphere.

    Parameters
    ----------
    points : (N, 3) float32

    Returns
    -------
    (N, 3) float32
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N,3) points, got shape {points.shape}")

    centroid = points.mean(axis=0, keepdims=True)
    pts = points - centroid
    dists = np.linalg.norm(pts, axis=1)
    max_dist = float(dists.max()) if pts.shape[0] > 0 else 0.0
    if max_dist > 0:
        pts = pts / max_dist
    return pts.astype(np.float32)


def random_rotate_z(points: np.ndarray) -> np.ndarray:
    """Random rotation around Z axis (vertical)."""
    theta = np.random.uniform(0.0, 2.0 * math.pi)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return points @ R.T


def jitter_points(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """Add small Gaussian noise to points."""
    noise = np.clip(np.random.normal(0.0, sigma, size=points.shape), -clip, clip)
    return (points + noise).astype(np.float32)


def augment_points(points: np.ndarray) -> np.ndarray:
    pts = random_rotate_z(points)
    pts = jitter_points(pts)
    return pts


def list_cluster_pcds(cluster_dir: str) -> List[str]:
    patterns = [
        "config*_cluster_*.pcd",
        "*.pcd",
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(cluster_dir, pat)))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(
            f"No .pcd cluster files found in {cluster_dir}. "
            "Run scripts/run_pipeline.sh first to generate clusters."
        )
    return files


def pointcloud_generator(
    files: List[str], num_points: int, augment: bool = True
):  # pragma: no cover - training-time
    """Yield normalized (num_points, 3) point clouds sampled from cluster PCDs."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")

    while True:
        np.random.shuffle(files)
        for f in files:
            coords = load_xyz_from_pcd(f)
            if coords.shape[0] == 0:
                continue

            if coords.shape[0] < num_points:
                idx = np.random.choice(coords.shape[0], num_points, replace=True)
            else:
                idx = np.random.choice(coords.shape[0], num_points, replace=False)

            pts = coords[idx]
            pts = normalize_points(pts)
            if augment:
                pts = augment_points(pts)
            yield pts.astype(np.float32)


class TransformerBlock(tf.keras.layers.Layer):  # pragma: no cover - pure TF
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(int(d_model * mlp_ratio), activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_output = self.mha(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        return out2


class PointEmbedding(tf.keras.layers.Layer):  # pragma: no cover - pure TF
    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_mlp = tf.keras.layers.Dense(d_model, activation="relu")
        self.pos_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_model, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )

    def call(self, coords: tf.Tensor) -> tf.Tensor:
        base = self.feature_mlp(coords)
        pos = self.pos_mlp(coords)
        return base + pos


def build_point_transformer_autoencoder(
    num_points: int,
    d_model: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    latent_dim: int = 256,
) -> tf.keras.Model:  # pragma: no cover - pure TF
    """Build a transformer-based autoencoder for 3D point clouds.

    Input:  (B, num_points, 3)
    Output: (B, num_points, 3) reconstructed, normalized coordinates.
    """
    inputs = tf.keras.Input(shape=(num_points, 3), name="points")

    # Encoder
    x = PointEmbedding(d_model, name="point_embed")(inputs)
    for i in range(num_layers):
        x = TransformerBlock(d_model=d_model, num_heads=num_heads, name=f"enc_block_{i+1}")(x)

    x_pooled = tf.keras.layers.GlobalAveragePooling1D(name="global_pool")(x)
    latent = tf.keras.layers.Dense(latent_dim, activation="relu", name="latent")(x_pooled)

    # Decoder: repeat latent across points and run transformer blocks again
    x_dec = tf.keras.layers.RepeatVector(num_points, name="repeat_latent")(latent)
    x_dec = tf.keras.layers.Dense(d_model, activation="relu", name="dec_input_proj")(x_dec)
    for i in range(num_layers):
        x_dec = TransformerBlock(d_model=d_model, num_heads=num_heads, name=f"dec_block_{i+1}")(x_dec)

    outputs = tf.keras.layers.Dense(3, name="recon_coords")(x_dec)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="pointcloud_transformer_autoencoder")
    return model


def build_dataset(
    cluster_dir: str,
    num_points: int,
    batch_size: int,
    augment: bool = True,
) -> tuple[tf.data.Dataset, int]:  # pragma: no cover - training-time
    files = list_cluster_pcds(cluster_dir)
    gen = lambda: pointcloud_generator(files, num_points=num_points, augment=augment)

    output_signature = tf.TensorSpec(shape=(num_points, 3), dtype=tf.float32)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    # For an autoencoder, the reconstruction target is the input itself.
    # Keras 3/optree does not support None targets in the data pipeline,
    # so we explicitly create (x, x) pairs.
    ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, len(files)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Train a transformer-based autoencoder on vineyard point cloud clusters. "
            "Uses .pcd clusters from scripts/out_cluster (output of run_pipeline.sh)."
        )
    )
    ap.add_argument(
        "--cluster-dir",
        type=str,
        default="./out_cluster",
        help="Directory containing cluster .pcd files (default: ./out_cluster)",
    )
    ap.add_argument(
        "--num-points",
        type=int,
        default=2048,
        help="Number of points to sample per cluster (default: 2048)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    ap.add_argument(
        "--steps-per-epoch",
        type=int,
        default=200,
        help="Steps per epoch (batches). Set <=0 to auto-compute from number of clusters.",
    )
    ap.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Transformer feature dimension (default: 128)",
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer blocks in encoder/decoder (default: 4)",
    )
    ap.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    ap.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent feature dimension (default: 256)",
    )
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="./ae_checkpoints",
        help="Directory to store checkpoints and logs (default: ./ae_checkpoints)",
    )
    return ap.parse_args()


def configure_gpu() -> None:  # pragma: no cover - runtime env
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[WARN] No GPU found. Training will run on CPU and be slow.")
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Using {len(gpus)} GPU(s): {[g.name for g in gpus]}")
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[WARN] Failed to configure GPU memory growth: {exc}")


def main() -> None:  # pragma: no cover - entry point
    args = parse_args()

    configure_gpu()

    cluster_dir = args.cluster_dir
    num_points = args.num_points
    batch_size = args.batch_size

    print(f"[INFO] Building dataset from cluster dir: {cluster_dir}")
    ds, num_files = build_dataset(
        cluster_dir=cluster_dir,
        num_points=num_points,
        batch_size=batch_size,
        augment=True,
    )
    print(f"[INFO] Found {num_files} cluster PCD files.")

    print(
        f"[INFO] Building model: num_points={num_points}, d_model={args.d_model}, "
        f"num_layers={args.num_layers}, num_heads={args.num_heads}, latent_dim={args.latent_dim}"
    )
    model = build_point_transformer_autoencoder(
        num_points=num_points,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "best_model.keras"
    log_dir = output_dir / "logs"

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="loss",
            save_best_only=True,
            save_weights_only=False,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]

    if args.steps_per_epoch <= 0:
        steps_per_epoch = max(1, num_files // max(1, batch_size))
    else:
        steps_per_epoch = args.steps_per_epoch

    print(
        f"[INFO] Starting training: epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, "
        f"batch_size={batch_size}"
    )
    model.fit(
        ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    final_model_path = output_dir / "final_model.keras"
    model.save(str(final_model_path))
    print(f"[INFO] Saved final model to {final_model_path}")
    print(f"[INFO] Best checkpoint saved to {ckpt_path}")


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
