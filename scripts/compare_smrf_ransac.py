"""Run SMRF twice (default vs tuned PDAL params) and a single-plane RANSAC fit
on the same input LAS, then save thesis-ready comparison figures.

Reuses smrf_ground_classification.run_smrf_classification /
split_ground_non_ground for the PDAL pipeline logic.
"""

import argparse
import sys
from pathlib import Path

import laspy
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from smrf_ground_classification import (
    run_smrf_classification,
    split_ground_non_ground,
)


DEFAULT_PARAMS = {"slope": 0.60, "window": 20.0, "threshold": 2.0, "scalar": 2.00}
TUNED_PARAMS = {"slope": 0.15, "window": 16.0, "threshold": 0.5, "scalar": 1.25}


def _load_xyz(las_path: str) -> np.ndarray:
    with laspy.open(las_path) as src:
        pts = src.read()
    return np.vstack((pts.x, pts.y, pts.z)).T.astype(np.float64)


def _subsample(xyz: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    rng = np.random.default_rng(seed)
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]


def _smrf_run(input_las: str, out_dir: Path, label: str, params: dict) -> str:
    sub = out_dir / label
    sub.mkdir(parents=True, exist_ok=True)
    classified = run_smrf_classification(input_las, out_dir=str(sub), smrf_params=params)
    _, non_ground = split_ground_non_ground(classified, str(sub))
    return non_ground


def plot_smrf_comparison(
    ng_default: np.ndarray,
    ng_tuned: np.ndarray,
    out_path: Path,
    plot_max_points: int = 400_000,
) -> None:
    d = _subsample(ng_default, plot_max_points, seed=1)
    t = _subsample(ng_tuned, plot_max_points, seed=2)

    z_all = np.concatenate([d[:, 2], t[:, 2]])
    vmin, vmax = float(np.percentile(z_all, 1)), float(np.percentile(z_all, 99))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=200)

    for ax, pts in ((axes[0], d), (axes[1], t)):
        sc = ax.scatter(
            pts[:, 0], pts[:, 1], c=pts[:, 2],
            cmap="viridis", s=0.4, vmin=vmin, vmax=vmax, linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        plt.colorbar(sc, ax=ax, label="Z [m]", shrink=0.85)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_ransac_failure(
    full_xyz: np.ndarray,
    out_path: Path,
    distance_threshold: float = 0.30,
    ransac_n: int = 3,
    num_iterations: int = 1000,
    fit_max_points: int = 1_500_000,
    plot_max_points: int = 500_000,
) -> None:
    fit_pts = _subsample(full_xyz, fit_max_points, seed=3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fit_pts)

    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    a, b, c, d = plane_model
    print(f"RANSAC plane: {a:.4f} x + {b:.4f} y + {c:.4f} z + {d:.4f} = 0")

    inlier_mask = np.zeros(fit_pts.shape[0], dtype=bool)
    inlier_mask[np.asarray(inlier_idx, dtype=np.int64)] = True

    inliers = fit_pts[inlier_mask]
    outliers = fit_pts[~inlier_mask]

    in_plot = _subsample(inliers, plot_max_points // 2, seed=4)
    out_plot = _subsample(outliers, plot_max_points // 2, seed=5)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=200)
    ax.scatter(in_plot[:, 0], in_plot[:, 1], c="0.55", s=0.4, linewidths=0,
               label=f"RANSAC inliers (plane / 'ground'): {inliers.shape[0]:,}")
    ax.scatter(out_plot[:, 0], out_plot[:, 1], c="#2ca02c", s=0.4, linewidths=0,
               label=f"RANSAC outliers ('above'): {outliers.shape[0]:,}")
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    leg = ax.legend(loc="upper right", markerscale=8, framealpha=0.9)
    for h in leg.legend_handles:
        h.set_sizes([30])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="../../datasource/2025-07-15-MS_Vinograd_1.las",
                   help="Path to input LAS/LAZ file.")
    p.add_argument("--out-dir", default="./out_ground_compare",
                   help="Directory for SMRF intermediate outputs.")
    p.add_argument("--images-dir", default="../images",
                   help="Directory where the comparison PNGs are written.")
    p.add_argument("--ransac-dist", type=float, default=0.30,
                   help="RANSAC distance threshold in metres.")
    args = p.parse_args(argv)

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=== [1/3] SMRF default ===")
    ng_default_path = _smrf_run(str(input_path), out_dir, "default", DEFAULT_PARAMS)

    print("=== [2/3] SMRF tuned ===")
    ng_tuned_path = _smrf_run(str(input_path), out_dir, "tuned", TUNED_PARAMS)

    ng_default_xyz = _load_xyz(ng_default_path)
    ng_tuned_xyz = _load_xyz(ng_tuned_path)

    plot_smrf_comparison(
        ng_default_xyz, ng_tuned_xyz, images_dir / "smrf_comparison.png",
    )

    print("=== [3/3] RANSAC single plane ===")
    full_xyz = _load_xyz(str(input_path))
    plot_ransac_failure(
        full_xyz, images_dir / "ransac_failure.png",
        distance_threshold=args.ransac_dist,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
