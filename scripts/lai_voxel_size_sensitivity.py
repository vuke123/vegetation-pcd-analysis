#!/usr/bin/env python3
"""
LAI proxy vs voxel size sensitivity analysis.

Sweeps the voxel edge length over a configurable range, recomputes the
segment-based canopy LAI proxy for each cluster LAS, and plots
LAI_proxy_mean against voxel_size to visualise the dependency.

Why this script exists
----------------------
The Beer-Lambert LAI proxy in `compute_canopy_structure.py` inverts a
gap fraction that is derived from a voxel occupancy grid. The voxel
edge length is therefore a hyperparameter that *materially* shifts the
reported LAI value:

  - Too small (e.g. ~2 cm at this point density): most voxels touch <= 1
    return, the (v, z) plane reads as mostly empty, gap fraction is
    artificially high and LAI is suppressed.
  - Too large (e.g. ~30 cm): every (v, z) cell is occupied, gap fraction
    collapses to ~0, LAI explodes towards `-log(EPS)/G` (a numeric ceiling).

A correctly chosen voxel should sit in a *stable* portion of the curve,
where small perturbations in voxel size change LAI only slightly.

What this script does
---------------------
1. Picks a small set of representative cluster LAS files (default: a few
   from the existing out_cluster_las folder; --las-files overrides).
2. For each voxel size in --voxel-sizes, calls the same per-cluster
   processing function used by the production pipeline
   (`process_cluster` in `compute_canopy_structure.py`). This means the
   numbers shown here are the same numbers the pipeline would produce.
3. Aggregates a single LAI_proxy_mean per (cluster, voxel_size) and
   plots one line per cluster, plus the across-clusters mean.
4. Saves a PNG line graph and (optionally) the underlying data as a CSV.

Defaults are tuned to be cheap to run (~10-20s per cluster at the
default sweep).

Examples
--------
  # Default sweep with default clusters:
  python3 lai_voxel_size_sensitivity.py

  # Custom sweep + custom clusters:
  python3 lai_voxel_size_sensitivity.py \\
      --voxel-sizes 0.03 0.05 0.08 0.10 0.15 0.20 0.30 \\
      --las-files /path/to/cluster_01_ndvi.las /path/to/cluster_04_ndvi.las

  # Use the in-repo (vegetation-pcd-analysis) folder:
  python3 lai_voxel_size_sensitivity.py --las-dir scripts/out_cluster_las
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the production segment-based pipeline so the LAI values reported
# here are exactly what compute_canopy_structure.py would produce.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compute_canopy_structure import process_cluster  # noqa: E402


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VOXEL_SIZES = [0.03, 0.05, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30]
DEFAULT_SEGMENT_LENGTH = 1.0

# Default clusters to sweep. We pick a small varied set (different
# cluster geometries). Override via --las-files or --las-dir.
DEFAULT_CLUSTER_BASENAMES = [
    "config1_leaf00cm_tol40cm_cluster_01_ndvi.las",
    "config1_leaf00cm_tol40cm_cluster_03_ndvi.las",
    "config1_leaf00cm_tol40cm_cluster_04_ndvi.las",
    "config1_leaf00cm_tol40cm_cluster_06_ndvi.las",
]


# ---------------------------------------------------------------------------
# Cluster discovery
# ---------------------------------------------------------------------------

def _resolve_default_data_dir(repo_root: Path) -> Path:
    """Pick the most likely data folder.

    Preference order:
      1. ../scripts/out_cluster_las (sibling folder, currently the
         working data with non-zero NIR).
      2. <repo_root>/scripts/out_cluster_las.
    """
    sibling = repo_root.parent / "scripts" / "out_cluster_las"
    if sibling.is_dir():
        return sibling
    return repo_root / "scripts" / "out_cluster_las"


def _discover_clusters(args, repo_root: Path) -> list[Path]:
    if args.las_files:
        files = [Path(p).expanduser().resolve() for p in args.las_files]
    else:
        data_dir = Path(args.las_dir).expanduser().resolve() if args.las_dir \
            else _resolve_default_data_dir(repo_root)
        if not data_dir.is_dir():
            raise SystemExit(f"Data directory does not exist: {data_dir}")
        files = [data_dir / name for name in DEFAULT_CLUSTER_BASENAMES]
        files = [f for f in files if f.is_file()]
        if not files:
            # Fall back to whatever cluster_*.las files we can find.
            files = sorted(data_dir.glob("*_cluster_*_ndvi.las"))[:4]
    missing = [f for f in files if not f.is_file()]
    if missing:
        for m in missing:
            logger.error("Missing cluster LAS: %s", m)
        raise SystemExit(1)
    return files


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def _run_sweep(cluster_files: list[Path],
               voxel_sizes: list[float],
               segment_length: float) -> pd.DataFrame:
    """Run process_cluster for every (cluster, voxel_size) pair.

    Returns a long-form DataFrame with one row per (cluster, voxel_size).
    """
    rows: list[dict] = []
    for fp in cluster_files:
        for vs in voxel_sizes:
            t0 = time.perf_counter()
            try:
                result = process_cluster(fp, vs, segment_length)
            except Exception as exc:
                logger.error("FAILED voxel=%.3f cluster=%s : %s", vs, fp.name, exc)
                continue
            dt = time.perf_counter() - t0
            if result is None:
                logger.warning("Skipped voxel=%.3f cluster=%s", vs, fp.name)
                continue
            rows.append({
                "cluster_file": fp.name,
                "row_id": result.get("row_id"),
                "voxel_size": vs,
                "segment_length": segment_length,
                "n_points": result.get("point_count"),
                "n_segments_valid": result.get("n_segments_valid"),
                "lai_proxy_mean": result.get("lai_proxy_mean"),
                "lai_proxy_p50": result.get("lai_proxy_p50"),
                "lai_proxy_p90": result.get("lai_proxy_p90"),
                "gap_fraction_mean": result.get("gap_fraction_mean"),
                "porosity_mean": result.get("porosity_mean"),
                "elapsed_sec": dt,
            })
            logger.info("  voxel=%.3f m  cluster=%-50s  LAI_mean=%.3f  (%.1fs)",
                        vs, fp.name, result.get("lai_proxy_mean", float("nan")), dt)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_sweep(df: pd.DataFrame, out_path: Path,
                show_secondary: bool = True) -> None:
    """Save the LAI-vs-voxel-size line graph.

    Lines: one per cluster, plus a thick across-clusters mean curve.
    Optional second panel: gap fraction (which drives the LAI value) on
    the same x-axis, so the cause of the LAI curve is visible.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2 if show_secondary else 1, 1,
        figsize=(11, 8 if show_secondary else 5),
        sharex=True,
    )
    if not show_secondary:
        axes = [axes]

    # ------- top: LAI proxy -------
    ax = axes[0]
    cluster_files = sorted(df["cluster_file"].unique())
    cmap = plt.get_cmap("tab10")
    for i, fname in enumerate(cluster_files):
        sub = df[df["cluster_file"] == fname].sort_values("voxel_size")
        ax.plot(sub["voxel_size"].to_numpy(), sub["lai_proxy_mean"].to_numpy(),
                marker="o", lw=1.4, color=cmap(i % 10),
                label=fname.replace("config1_leaf00cm_tol40cm_", ""))

    # Mean curve across clusters at each voxel size.
    agg = df.groupby("voxel_size", as_index=False)["lai_proxy_mean"].mean()
    agg = agg.sort_values("voxel_size")
    ax.plot(agg["voxel_size"].to_numpy(), agg["lai_proxy_mean"].to_numpy(),
            marker="s", lw=2.6, color="black",
            label="mean across clusters")

    ax.set_ylabel("LAI proxy (segment mean)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    # ------- bottom: gap fraction (driver) -------
    if show_secondary:
        ax2 = axes[1]
        for i, fname in enumerate(cluster_files):
            sub = df[df["cluster_file"] == fname].sort_values("voxel_size")
            ax2.plot(sub["voxel_size"].to_numpy(),
                     sub["gap_fraction_mean"].to_numpy(),
                     marker="o", lw=1.2, color=cmap(i % 10), alpha=0.85)
        agg_gf = df.groupby("voxel_size", as_index=False)["gap_fraction_mean"].mean()
        agg_gf = agg_gf.sort_values("voxel_size")
        ax2.plot(agg_gf["voxel_size"].to_numpy(),
                 agg_gf["gap_fraction_mean"].to_numpy(),
                 marker="s", lw=2.6, color="black")
        ax2.set_ylabel("Gap fraction (segment mean)")
        ax2.set_xlabel("Voxel size [m]")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.0, 1.0)
    else:
        axes[0].set_xlabel("Voxel size [m]")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    logger.info("Saved figure to %s", out_path)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_pivot(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(
        index="cluster_file",
        columns="voxel_size",
        values="lai_proxy_mean",
        aggfunc="first",
    )
    print()
    print("LAI proxy (segment mean) — rows = cluster, columns = voxel size [m]")
    print("=" * 100)
    with pd.option_context("display.float_format", lambda x: f"{x:6.3f}",
                           "display.width", 220):
        print(pivot.to_string())
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_out = repo_root / "images" / "lai_vs_voxel_size.png"
    default_csv = repo_root / "images" / "lai_vs_voxel_size.csv"

    ap = argparse.ArgumentParser(
        description="LAI-proxy vs voxel-size sensitivity sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--voxel-sizes", type=float, nargs="+",
                    default=DEFAULT_VOXEL_SIZES,
                    help=f"Voxel sizes in metres (default: {DEFAULT_VOXEL_SIZES})")
    ap.add_argument("--segment-length", type=float, default=DEFAULT_SEGMENT_LENGTH,
                    help=f"Segment length [m] (default: {DEFAULT_SEGMENT_LENGTH})")
    ap.add_argument("--las-files", nargs="*", default=None,
                    help="Explicit list of cluster LAS files to sweep")
    ap.add_argument("--las-dir", default=None,
                    help="Directory of cluster LAS files (overrides default discovery)")
    ap.add_argument("--out", type=Path, default=default_out,
                    help=f"Output figure path (default: {default_out})")
    ap.add_argument("--csv", type=Path, default=default_csv,
                    help=f"Output CSV of sweep data (default: {default_csv})")
    ap.add_argument("--no-secondary", action="store_true",
                    help="Skip the gap-fraction sub-plot")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip plot, only write CSV and print pivot")
    ap.add_argument("--verbose", action="store_true",
                    help="Verbose logging")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if any(v <= 0 for v in args.voxel_sizes):
        raise SystemExit("All voxel sizes must be > 0")
    if args.segment_length <= 0:
        raise SystemExit("Segment length must be > 0")

    cluster_files = _discover_clusters(args, repo_root)
    logger.info("Sweeping %d voxel sizes × %d clusters = %d runs",
                len(args.voxel_sizes), len(cluster_files),
                len(args.voxel_sizes) * len(cluster_files))
    for f in cluster_files:
        logger.info("  cluster: %s", f)

    df = _run_sweep(cluster_files,
                    sorted(args.voxel_sizes),
                    args.segment_length)
    if df.empty:
        raise SystemExit("No results produced.")

    # Persist sweep data
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    logger.info("Wrote sweep data to %s", args.csv)

    _print_pivot(df)

    if not args.no_plot:
        _plot_sweep(df, args.out, show_secondary=not args.no_secondary)


if __name__ == "__main__":
    main()
