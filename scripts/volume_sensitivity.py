#!/usr/bin/env python3
"""
Volume sensitivity sweeps for the two canopy-volume estimators used by the
vineyard pipeline.

Two independent sweeps are produced, each varying the parameter that the
underlying estimator actually depends on:

  1. Voxel-based volume  (compute_voxel_volume)
       X-axis: voxel_size  [m]
       Y-axis: vol_voxel   [m^3]
       Curve drivers: V = n_unique_voxels(voxel_size) * voxel_size^3.
       As voxel_size shrinks every return falls in its own voxel, so the
       grid undercounts canopy; as voxel_size grows few voxels cover the
       whole bounding box, so the grid overcounts.

  2. Slice + alpha-shape volume  (compute_slice_volume)
       X-axis: rmax  [m]   (alpha-shape circumradius threshold)
       Y-axis: vol_slice [m^3]
       Curve drivers: each Z slice's alpha hull tightens with small rmax
       (concave, less area, less volume) and relaxes toward the convex
       hull as rmax grows (more area, more volume, plateau).

Outputs (default paths under ../images/):
  - volume_vs_voxel_size.png
  - volume_vs_alpha_radius.png
  - volume_vs_voxel_size.csv
  - volume_vs_alpha_radius.csv

Both estimators are imported directly from compute_row_features.py so the
numbers shown here are identical to the ones the production pipeline (and
therefore the vineyard_app GUI) report.

Examples
--------
  # Default sweep on all cluster_*_ndvi.las files in out_cluster_las/:
  python3 volume_sensitivity.py

  # Custom counts and ranges:
  python3 volume_sensitivity.py \\
      --n-voxel-sizes 30 --voxel-range 0.005 0.6 \\
      --n-rmax 30       --rmax-range 0.05 3.0

  # Restrict to a single cluster:
  python3 volume_sensitivity.py \\
      --las-files out_cluster_las/config1_leaf00cm_tol40cm_cluster_03_ndvi.las
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compute_row_features import (  # noqa: E402
    compute_voxel_volume,
    compute_slice_volume,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VOXEL_RANGE = (0.01, 0.50)   # metres
DEFAULT_N_VOXEL_SIZES = 25
DEFAULT_RMAX_RANGE = (0.05, 3.00)    # metres
DEFAULT_N_RMAX = 25
DEFAULT_N_SLICES = 30                # matches compute_row_features.compute_slice_volume default


# ---------------------------------------------------------------------------
# Cluster discovery
# ---------------------------------------------------------------------------

def _default_data_dir() -> Path:
    return HERE / "out_cluster_las"


def _discover_clusters(args) -> list[Path]:
    if args.las_files:
        files = [Path(p).expanduser().resolve() for p in args.las_files]
    else:
        data_dir = Path(args.las_dir).expanduser().resolve() if args.las_dir \
            else _default_data_dir()
        if not data_dir.is_dir():
            raise SystemExit(f"Data directory does not exist: {data_dir}")
        files = sorted(data_dir.glob("*_cluster_*_ndvi.las"))
    if not files:
        raise SystemExit("No cluster LAS files found.")
    missing = [f for f in files if not f.is_file()]
    if missing:
        for m in missing:
            logger.error("Missing cluster LAS: %s", m)
        raise SystemExit(1)
    return files


def _load_points(las_path: Path) -> np.ndarray:
    las = laspy.read(str(las_path))
    return np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

def _log_space(lo: float, hi: float, n: int) -> np.ndarray:
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError(f"Invalid range [{lo}, {hi}] for log-spaced sweep")
    return np.geomspace(lo, hi, n)


def _sweep_voxel_volume(cluster_points: dict[str, np.ndarray],
                        voxel_sizes: np.ndarray) -> pd.DataFrame:
    """vol_voxel for each (cluster, voxel_size)."""
    rows: list[dict] = []
    for name, pts in cluster_points.items():
        for vs in voxel_sizes:
            t0 = time.perf_counter()
            try:
                res = compute_voxel_volume(pts, voxel_size=float(vs))
            except Exception as exc:
                logger.error("FAILED voxel=%.4f cluster=%s : %s", vs, name, exc)
                continue
            dt = time.perf_counter() - t0
            rows.append({
                "cluster_file": name,
                "voxel_size": float(vs),
                "vol_voxel": res["vol_voxel"],
                "n_voxels": res["n_voxels"],
                "n_points": int(len(pts)),
                "elapsed_sec": dt,
            })
            logger.debug("  voxel=%.4f m  cluster=%-50s  V=%.3f m^3  (%.2fs)",
                         vs, name, res["vol_voxel"], dt)
    return pd.DataFrame(rows)


def _sweep_slice_volume(cluster_points: dict[str, np.ndarray],
                        rmax_values: np.ndarray,
                        n_slices: int) -> pd.DataFrame:
    """vol_slice for each (cluster, rmax)."""
    rows: list[dict] = []
    for name, pts in cluster_points.items():
        for r in rmax_values:
            t0 = time.perf_counter()
            try:
                res = compute_slice_volume(pts, n_slices=n_slices, rmax=float(r))
            except Exception as exc:
                logger.error("FAILED rmax=%.4f cluster=%s : %s", r, name, exc)
                continue
            dt = time.perf_counter() - t0
            rows.append({
                "cluster_file": name,
                "rmax": float(r),
                "n_slices": n_slices,
                "vol_slice": res["vol_slice"],
                "n_slices_used": res["n_slices_used"],
                "n_points": int(len(pts)),
                "elapsed_sec": dt,
            })
            logger.debug("  rmax=%.4f m  cluster=%-50s  V=%.3f m^3  (%.2fs)",
                         r, name, res["vol_slice"], dt)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sweep(df: pd.DataFrame,
                x_col: str,
                y_col: str,
                out_path: Path,
                title: str,
                x_label: str,
                y_label: str) -> None:
    """One line per cluster + a thick mean curve. Log-x axis (sweeps are log-spaced)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_files = sorted(df["cluster_file"].unique())
    cmap = plt.get_cmap("tab10")

    for i, fname in enumerate(cluster_files):
        sub = df[df["cluster_file"] == fname].sort_values(x_col)
        ax.plot(sub[x_col].to_numpy(), sub[y_col].to_numpy(),
                marker="o", lw=1.2, color=cmap(i % 10),
                alpha=0.85,
                label=fname.replace("config1_leaf00cm_tol40cm_", "").replace("_ndvi.las", ""))

    agg = (
        df.groupby(x_col, as_index=False)[y_col]
          .mean()
          .sort_values(x_col)
    )
    ax.plot(agg[x_col].to_numpy(), agg[y_col].to_numpy(),
            marker="s", lw=2.6, color="black",
            label="mean across clusters")

    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_pivot(df: pd.DataFrame, x_col: str, y_col: str, label: str) -> None:
    pivot = df.pivot_table(
        index="cluster_file",
        columns=x_col,
        values=y_col,
        aggfunc="first",
    )
    print()
    print(f"{label} — rows = cluster, columns = {x_col}")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:8.3f}",
                           "display.width", 240):
        print(pivot.to_string())
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = HERE.parent
    default_out_dir = repo_root / "images"

    ap = argparse.ArgumentParser(
        description="Volume vs voxel_size / alpha-radius sensitivity sweeps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--las-files", nargs="*", default=None,
                    help="Explicit list of cluster LAS files to sweep.")
    ap.add_argument("--las-dir", default=None,
                    help="Directory of cluster LAS files (default: ./out_cluster_las).")
    ap.add_argument("--n-voxel-sizes", type=int, default=DEFAULT_N_VOXEL_SIZES,
                    help=f"Number of voxel sizes in the sweep (default: {DEFAULT_N_VOXEL_SIZES}).")
    ap.add_argument("--voxel-range", type=float, nargs=2, default=DEFAULT_VOXEL_RANGE,
                    metavar=("LO", "HI"),
                    help=f"Voxel size range in metres (default: {DEFAULT_VOXEL_RANGE}).")
    ap.add_argument("--n-rmax", type=int, default=DEFAULT_N_RMAX,
                    help=f"Number of alpha-radius values in the sweep (default: {DEFAULT_N_RMAX}).")
    ap.add_argument("--rmax-range", type=float, nargs=2, default=DEFAULT_RMAX_RANGE,
                    metavar=("LO", "HI"),
                    help=f"Alpha-radius range in metres (default: {DEFAULT_RMAX_RANGE}).")
    ap.add_argument("--n-slices", type=int, default=DEFAULT_N_SLICES,
                    help=f"Z-slice count for the slice volume (default: {DEFAULT_N_SLICES}).")
    ap.add_argument("--out-dir", type=Path, default=default_out_dir,
                    help=f"Directory for PNG/CSV outputs (default: {default_out_dir}).")
    ap.add_argument("--skip-voxel", action="store_true",
                    help="Skip the voxel-size sweep.")
    ap.add_argument("--skip-slice", action="store_true",
                    help="Skip the alpha-radius sweep.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip plot generation (CSV only).")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.skip_voxel and args.skip_slice:
        raise SystemExit("Both sweeps skipped — nothing to do.")

    cluster_files = _discover_clusters(args)
    logger.info("Found %d cluster LAS files", len(cluster_files))

    # Load every cluster once. The point clouds for vine rows are small enough
    # (~1e4–1e5 points each) that holding them in memory is cheaper than
    # re-reading the LAS for every (cluster, param) pair.
    cluster_points: dict[str, np.ndarray] = {}
    for fp in cluster_files:
        pts = _load_points(fp)
        if len(pts) < 10:
            logger.warning("Skipping %s (only %d points)", fp.name, len(pts))
            continue
        cluster_points[fp.name] = pts
        logger.info("  loaded %-55s  %d points", fp.name, len(pts))
    if not cluster_points:
        raise SystemExit("No usable cluster point clouds.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Sweep 1: voxel-based volume ---
    if not args.skip_voxel:
        vs = _log_space(args.voxel_range[0], args.voxel_range[1], args.n_voxel_sizes)
        logger.info("Voxel sweep: %d log-spaced sizes from %.4f to %.4f m",
                    args.n_voxel_sizes, vs[0], vs[-1])
        df_v = _sweep_voxel_volume(cluster_points, vs)
        csv_v = args.out_dir / "volume_vs_voxel_size.csv"
        df_v.to_csv(csv_v, index=False)
        logger.info("Wrote %s", csv_v)
        _print_pivot(df_v, "voxel_size", "vol_voxel",
                     "vol_voxel [m^3]")
        if not args.no_plot:
            _plot_sweep(
                df_v,
                x_col="voxel_size",
                y_col="vol_voxel",
                out_path=args.out_dir / "volume_vs_voxel_size.png",
                title="Voxel-based volume vs voxel size",
                x_label="voxel size [m]",
                y_label="vol_voxel [m³]",
            )

    # --- Sweep 2: slice + alpha-shape volume ---
    if not args.skip_slice:
        rs = _log_space(args.rmax_range[0], args.rmax_range[1], args.n_rmax)
        logger.info("Alpha-radius sweep: %d log-spaced rmax from %.4f to %.4f m  (n_slices=%d)",
                    args.n_rmax, rs[0], rs[-1], args.n_slices)
        df_s = _sweep_slice_volume(cluster_points, rs, args.n_slices)
        csv_s = args.out_dir / "volume_vs_alpha_radius.csv"
        df_s.to_csv(csv_s, index=False)
        logger.info("Wrote %s", csv_s)
        _print_pivot(df_s, "rmax", "vol_slice",
                     "vol_slice [m^3]")
        if not args.no_plot:
            _plot_sweep(
                df_s,
                x_col="rmax",
                y_col="vol_slice",
                out_path=args.out_dir / "volume_vs_alpha_radius.png",
                title=f"Slice + alpha-shape volume vs alpha radius (n_slices={args.n_slices})",
                x_label="alpha radius rmax [m]",
                y_label="vol_slice [m³]",
            )


if __name__ == "__main__":
    main()
