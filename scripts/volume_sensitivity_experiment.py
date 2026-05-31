#!/usr/bin/env python3
"""
EXPERIMENTAL — convergence-focused volume sensitivity study.

This script is **separate** from volume_sensitivity.py and **does not modify
production code** (compute_row_features.py is read-only here). It exists to
expose convergence behaviour that the simpler sweep does not visually
surface:

  Voxel sweep — three views of the same data
    1. vol_voxel vs voxel_size                      (linear-x AND log-x)
    2. occupied n_voxels vs voxel_size              (log-log, slope diagnostic)
    3. dV/d(log s) vs voxel_size                    (linear-x AND log-x)
       The defensible voxel-size band is shaded:
       the contiguous range where dV/d(log s) sits in the lower quartile,
       i.e. where V(s) is least sensitive to a small change in s.

  Filled-voxel variant (defined locally, not added to production)
    For each Z-layer of thickness s, rasterise the (x, y) occupancy at
    resolution s into a 2D binary grid, run scipy.ndimage.binary_fill_holes,
    count filled cells, multiply by s^3 and sum across layers. The hole-
    filling closes interior gaps so the count saturates at the true canopy
    cross-section at fine s. Expected behaviour: a plateau at fine s,
    divergence only at coarse s (when the grid is too crude to capture
    canopy shape and overcounts).

  Alpha-shape asymptote
    rmax sweep extended (default 0.05 → 10.0 m). Per cluster, the limit is
    the slice-wise 2D **convex-hull** volume Σ A_convex,k · dz computed with
    the same n_slices used by compute_slice_volume. The convex-hull volume
    is overlaid as a dashed horizontal line per cluster so the plateau the
    alpha-shape converges to is explicit.

  Plot duplication
    Log-x compresses the plateau; linear-x compresses the elbow. Each plot
    is therefore emitted twice, in a sibling `_log` / `_linear` PNG pair.

Outputs are written to vegetation-pcd-analysis/images/volume_sensitivity_experiment/
to keep them separate from the production sensitivity outputs.

Run (uses vineyard_app's venv that already has shapely/scipy/laspy)
-----
  /home/luka/Coding/DIPLOMSKI-RAD/vineyard_app/backend/.venv/bin/python \\
    /home/luka/Coding/DIPLOMSKI-RAD/vegetation-pcd-analysis/scripts/volume_sensitivity_experiment.py
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
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Reuse the production estimators read-only.
from compute_row_features import (  # noqa: E402
    compute_voxel_volume,
    compute_slice_volume,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_VOXEL_RANGE = (0.01, 0.50)      # metres — voxel-volume sweep
DEFAULT_N_VOXEL_SIZES = 28

DEFAULT_VOXEL_FILLED_RANGE = (0.015, 0.50)  # filled-voxel sweep
DEFAULT_N_VOXEL_FILLED = 20                 # binary_fill_holes is heavier per s

DEFAULT_RMAX_RANGE = (0.05, 10.0)       # extended to expose plateau
DEFAULT_N_RMAX = 28

DEFAULT_N_SLICES = 30                    # match compute_slice_volume default


# ---------------------------------------------------------------------------
# Experimental volume estimator: voxel grid with per-slice hole filling
# ---------------------------------------------------------------------------

def compute_voxel_volume_filled(points: np.ndarray, voxel_size: float) -> dict:
    """
    Per-Z-slice 2D binary occupancy + hole filling, then count × s^3.

    Each Z-layer is one voxel thick (dz = s). Within a layer the (x, y)
    voxel cells touched by at least one point are marked True; interior
    holes are filled with scipy.ndimage.binary_fill_holes. Filled cells
    are counted and the count is multiplied by s^3 (cell volume) before
    summing across layers.

    Why this differs from compute_voxel_volume
    -----------------------------------------
    `compute_voxel_volume` counts unique occupied voxel cells only. A row of
    foliage with sparse interior points (typical for LiDAR/MS scans) has
    many empty interior cells, so its plain voxel count under-represents
    the canopy at fine s — and the s^3 multiplier blows up before the
    occupancy catches up. Filling the per-layer interior recovers the
    correct cross-sectional area, so the curve should plateau at fine s
    (the same canopy, just sampled more finely) and only diverge at coarse
    s when the layer grid itself is too coarse to resolve canopy shape.

    Returns
    -------
    dict with vol_voxel_filled, voxel_size, n_voxels_filled.
    """
    s = float(voxel_size)
    if s <= 0:
        raise ValueError(f"voxel_size must be > 0 (got {s})")

    min_b = points.min(axis=0)
    idx = np.floor((points - min_b) / s).astype(np.int64)
    nx = int(idx[:, 0].max()) + 1
    ny = int(idx[:, 1].max()) + 1

    # Group by Z index without holding the whole 3D grid.
    order = np.argsort(idx[:, 2], kind="stable")
    idx_sorted = idx[order]
    z_sorted = idx_sorted[:, 2]
    unique_z, starts = np.unique(z_sorted, return_index=True)
    starts = np.append(starts, len(z_sorted))

    total_cells = 0
    for j in range(len(unique_z)):
        a, b = starts[j], starts[j + 1]
        layer_xy = idx_sorted[a:b, :2]
        grid = np.zeros((nx, ny), dtype=bool)
        grid[layer_xy[:, 0], layer_xy[:, 1]] = True
        total_cells += int(binary_fill_holes(grid).sum())

    return {
        "vol_voxel_filled": float(total_cells * (s ** 3)),
        "voxel_size": s,
        "n_voxels_filled": int(total_cells),
    }


# ---------------------------------------------------------------------------
# Per-cluster 2D convex-hull slice volume baseline (alpha-shape asymptote)
# ---------------------------------------------------------------------------

def compute_slice_convex_volume(points: np.ndarray, n_slices: int = DEFAULT_N_SLICES) -> float:
    """
    Σ A_convex,k · dz over n_slices Z-bins. This is the upper limit that
    compute_slice_volume converges to as the alpha radius rmax → ∞ (every
    Delaunay triangle is kept, alpha shape ≡ convex hull).
    """
    z = points[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 0.01:
        return 0.0
    edges = np.linspace(zmin, zmax, n_slices + 1)
    total = 0.0
    for k in range(n_slices):
        lo, hi = edges[k], edges[k + 1]
        if k < n_slices - 1:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)
        s_pts = points[mask]
        if len(s_pts) < 3:
            continue
        try:
            hull = ConvexHull(s_pts[:, :2])
        except Exception:
            continue
        # ConvexHull.volume on 2D points gives the polygon area.
        total += float(hull.volume) * (hi - lo)
    return float(total)


# ---------------------------------------------------------------------------
# Cluster discovery / loading
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


def _log_space(lo: float, hi: float, n: int) -> np.ndarray:
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError(f"Invalid range [{lo}, {hi}] for log-spaced sweep")
    return np.geomspace(lo, hi, n)


def _short_name(fname: str) -> str:
    return fname.replace("config1_leaf00cm_tol40cm_", "").replace("_ndvi.las", "")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def _sweep_voxel(cluster_points: dict[str, np.ndarray],
                 voxel_sizes: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for name, pts in cluster_points.items():
        for vs in voxel_sizes:
            t0 = time.perf_counter()
            res = compute_voxel_volume(pts, voxel_size=float(vs))
            dt = time.perf_counter() - t0
            rows.append({
                "cluster_file": name,
                "voxel_size": float(vs),
                "vol_voxel": res["vol_voxel"],
                "n_voxels": res["n_voxels"],
                "n_points": int(len(pts)),
                "elapsed_sec": dt,
            })
    return pd.DataFrame(rows)


def _sweep_voxel_filled(cluster_points: dict[str, np.ndarray],
                        voxel_sizes: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    for name, pts in cluster_points.items():
        for vs in voxel_sizes:
            t0 = time.perf_counter()
            try:
                res = compute_voxel_volume_filled(pts, voxel_size=float(vs))
            except MemoryError as exc:
                logger.warning("OOM filled voxel=%.4f cluster=%s : %s", vs, name, exc)
                continue
            dt = time.perf_counter() - t0
            rows.append({
                "cluster_file": name,
                "voxel_size": float(vs),
                "vol_voxel_filled": res["vol_voxel_filled"],
                "n_voxels_filled": res["n_voxels_filled"],
                "n_points": int(len(pts)),
                "elapsed_sec": dt,
            })
            logger.debug("  filled voxel=%.4f cluster=%-45s V=%.3f m^3 (%.1fs)",
                         vs, name, res["vol_voxel_filled"], dt)
    return pd.DataFrame(rows)


def _sweep_alpha(cluster_points: dict[str, np.ndarray],
                 rmax_values: np.ndarray,
                 n_slices: int) -> pd.DataFrame:
    rows: list[dict] = []
    for name, pts in cluster_points.items():
        for r in rmax_values:
            t0 = time.perf_counter()
            res = compute_slice_volume(pts, n_slices=n_slices, rmax=float(r))
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
    return pd.DataFrame(rows)


def _convex_baseline(cluster_points: dict[str, np.ndarray],
                     n_slices: int) -> dict[str, float]:
    return {
        name: compute_slice_convex_volume(pts, n_slices=n_slices)
        for name, pts in cluster_points.items()
    }


# ---------------------------------------------------------------------------
# Defensible-band annotation
# ---------------------------------------------------------------------------

def _flattest_band(x_vals: np.ndarray, y_vals: np.ndarray,
                   quantile: float = 0.25) -> tuple[float, float] | None:
    """
    Find the contiguous range of x where dV/d(log x) lies in the lowest
    `quantile` of its values (i.e. where the curve is flattest per decade
    of x). Returns (x_lo, x_hi) or None.
    """
    if len(x_vals) < 3:
        return None
    logx = np.log(x_vals)
    dV_dlogs = np.gradient(y_vals, logx)
    thr = float(np.quantile(dV_dlogs, quantile))
    mask = dV_dlogs <= thr
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    # Pick the longest contiguous run (most credible flat region).
    splits = np.where(np.diff(idx) > 1)[0]
    runs = np.split(idx, splits + 1)
    best = max(runs, key=len)
    return float(x_vals[best[0]]), float(x_vals[best[-1]])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_axes(ax, x_scale: str, x_label: str, y_label: str, title: str) -> None:
    if x_scale == "log":
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def _plot_curves(df: pd.DataFrame,
                 x_col: str, y_col: str,
                 out_dir: Path,
                 stem: str,
                 title: str,
                 x_label: str, y_label: str,
                 horizontal_lines: dict[str, float] | None = None,
                 shade_x_range: tuple[float, float] | None = None,
                 shade_label: str | None = None) -> None:
    """
    Emit two PNGs (log-x and linear-x). One line per cluster + thick mean
    curve. Optional per-cluster horizontal dashed lines and an x-band shade.
    """
    import matplotlib.pyplot as plt

    cluster_files = sorted(df["cluster_file"].unique())
    cmap = plt.get_cmap("tab10")
    agg = (
        df.groupby(x_col, as_index=False)[y_col]
          .mean()
          .sort_values(x_col)
    )

    for x_scale in ("log", "linear"):
        fig, ax = plt.subplots(figsize=(10, 6))
        if shade_x_range is not None:
            lo, hi = shade_x_range
            ax.axvspan(lo, hi, color="green", alpha=0.12, zorder=0,
                       label=shade_label or "_nolegend_")

        for i, fname in enumerate(cluster_files):
            sub = df[df["cluster_file"] == fname].sort_values(x_col)
            ax.plot(sub[x_col].to_numpy(), sub[y_col].to_numpy(),
                    marker="o", lw=1.2, color=cmap(i % 10), alpha=0.85,
                    label=_short_name(fname))
            if horizontal_lines and fname in horizontal_lines:
                ax.axhline(horizontal_lines[fname],
                           color=cmap(i % 10), lw=1.0, ls="--", alpha=0.7)

        ax.plot(agg[x_col].to_numpy(), agg[y_col].to_numpy(),
                marker="s", lw=2.6, color="black", label="mean across clusters")

        _setup_axes(ax, x_scale, x_label, y_label, title)
        ax.legend(fontsize=7, loc="best", ncol=2)

        fig.tight_layout()
        out_path = out_dir / f"{stem}_{x_scale}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        logger.info("  saved %s", out_path)


def _plot_nvoxels_loglog(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_files = sorted(df["cluster_file"].unique())
    cmap = plt.get_cmap("tab10")
    for i, fname in enumerate(cluster_files):
        sub = df[df["cluster_file"] == fname].sort_values("voxel_size")
        ax.plot(sub["voxel_size"].to_numpy(), sub["n_voxels"].to_numpy(),
                marker="o", lw=1.2, color=cmap(i % 10), alpha=0.85,
                label=_short_name(fname))

    # Reference slope -3 line (the saturated regime: n_voxels ~ bbox / s^3).
    agg = df.groupby("voxel_size", as_index=False)["n_voxels"].mean()
    agg = agg.sort_values("voxel_size")
    xs = agg["voxel_size"].to_numpy()
    ys = agg["n_voxels"].to_numpy()
    # Pin the reference to the coarse-s end.
    anchor_idx = int(np.argmax(xs))
    ref_x = np.array([xs[0], xs[-1]])
    ref_y = ys[anchor_idx] * (xs[anchor_idx] / ref_x) ** 3
    ax.plot(ref_x, ref_y, color="grey", lw=1.0, ls=":", label="slope -3 reference")

    ax.plot(xs, ys, marker="s", lw=2.6, color="black", label="mean across clusters")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("voxel size [m]")
    ax.set_ylabel("occupied voxel count n_voxels")
    ax.set_title("Occupied voxel count vs voxel size (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    logger.info("  saved %s", out_path)


def _plot_dV_dlogs(df: pd.DataFrame, out_dir: Path, stem: str,
                   shade_x_range: tuple[float, float] | None) -> None:
    import matplotlib.pyplot as plt

    cluster_files = sorted(df["cluster_file"].unique())
    cmap = plt.get_cmap("tab10")
    # Build per-cluster derivative.
    deriv_rows = []
    for fname in cluster_files:
        sub = df[df["cluster_file"] == fname].sort_values("voxel_size")
        x = sub["voxel_size"].to_numpy()
        y = sub["vol_voxel"].to_numpy()
        d = np.gradient(y, np.log(x))
        deriv_rows.append((fname, x, d))

    agg = (
        df.groupby("voxel_size", as_index=False)["vol_voxel"]
          .mean()
          .sort_values("voxel_size")
    )
    x_mean = agg["voxel_size"].to_numpy()
    d_mean = np.gradient(agg["vol_voxel"].to_numpy(), np.log(x_mean))

    for x_scale in ("log", "linear"):
        fig, ax = plt.subplots(figsize=(10, 6))
        if shade_x_range is not None:
            lo, hi = shade_x_range
            ax.axvspan(lo, hi, color="green", alpha=0.12, zorder=0,
                       label="flattest band (lower-quartile dV/d log s)")
        for i, (fname, x, d) in enumerate(deriv_rows):
            ax.plot(x, d, marker="o", lw=1.2, color=cmap(i % 10),
                    alpha=0.85, label=_short_name(fname))
        ax.plot(x_mean, d_mean, marker="s", lw=2.6, color="black",
                label="mean across clusters")
        _setup_axes(
            ax, x_scale,
            "voxel size [m]",
            "dV_voxel / d(log s)  [m³ per natural-log decade of s]",
            "Voxel-volume log-derivative (small ⇒ V insensitive to s)",
        )
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}_{x_scale}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        logger.info("  saved %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = HERE.parent
    default_out_dir = repo_root / "images" / "volume_sensitivity_experiment"

    ap = argparse.ArgumentParser(
        description="Convergence-focused volume sensitivity experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--las-files", nargs="*", default=None)
    ap.add_argument("--las-dir", default=None)
    ap.add_argument("--n-voxel-sizes", type=int, default=DEFAULT_N_VOXEL_SIZES)
    ap.add_argument("--voxel-range", type=float, nargs=2, default=DEFAULT_VOXEL_RANGE,
                    metavar=("LO", "HI"))
    ap.add_argument("--n-voxel-filled", type=int, default=DEFAULT_N_VOXEL_FILLED)
    ap.add_argument("--voxel-filled-range", type=float, nargs=2,
                    default=DEFAULT_VOXEL_FILLED_RANGE, metavar=("LO", "HI"))
    ap.add_argument("--n-rmax", type=int, default=DEFAULT_N_RMAX)
    ap.add_argument("--rmax-range", type=float, nargs=2, default=DEFAULT_RMAX_RANGE,
                    metavar=("LO", "HI"))
    ap.add_argument("--n-slices", type=int, default=DEFAULT_N_SLICES)
    ap.add_argument("--out-dir", type=Path, default=default_out_dir)
    ap.add_argument("--skip-voxel", action="store_true")
    ap.add_argument("--skip-filled", action="store_true")
    ap.add_argument("--skip-alpha", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    cluster_files = _discover_clusters(args)
    logger.info("Found %d cluster LAS files", len(cluster_files))

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

    # ----- voxel sweep + diagnostics -----
    if not args.skip_voxel:
        vs = _log_space(args.voxel_range[0], args.voxel_range[1], args.n_voxel_sizes)
        logger.info("Voxel sweep: %d sizes from %.4f to %.4f m",
                    args.n_voxel_sizes, vs[0], vs[-1])
        df_v = _sweep_voxel(cluster_points, vs)
        df_v.to_csv(args.out_dir / "volume_vs_voxel_size.csv", index=False)

        # Flattest band derived from the cross-cluster mean curve.
        mean_v = (
            df_v.groupby("voxel_size", as_index=False)["vol_voxel"]
                .mean().sort_values("voxel_size")
        )
        band = _flattest_band(mean_v["voxel_size"].to_numpy(),
                              mean_v["vol_voxel"].to_numpy(),
                              quantile=0.25)
        if band is not None:
            logger.info("Defensible voxel band (lowest-quartile dV/d log s): [%.4f, %.4f] m",
                        band[0], band[1])

        _plot_curves(
            df_v, "voxel_size", "vol_voxel",
            args.out_dir, "volume_vs_voxel_size",
            "Voxel-based volume vs voxel size",
            "voxel size [m]", "vol_voxel [m³]",
            shade_x_range=band,
            shade_label="flattest band (lower-quartile dV/d log s)",
        )
        _plot_nvoxels_loglog(df_v, args.out_dir, "nvoxels_vs_voxel_size_loglog")
        _plot_dV_dlogs(df_v, args.out_dir, "dV_dlogs_vs_voxel_size",
                       shade_x_range=band)

    # ----- filled-voxel variant -----
    if not args.skip_filled:
        vsf = _log_space(args.voxel_filled_range[0],
                         args.voxel_filled_range[1],
                         args.n_voxel_filled)
        logger.info("Filled-voxel sweep: %d sizes from %.4f to %.4f m",
                    args.n_voxel_filled, vsf[0], vsf[-1])
        df_f = _sweep_voxel_filled(cluster_points, vsf)
        df_f.to_csv(args.out_dir / "volume_vs_voxel_size_filled.csv", index=False)
        if not df_f.empty:
            _plot_curves(
                df_f, "voxel_size", "vol_voxel_filled",
                args.out_dir, "volume_vs_voxel_size_filled",
                "Filled-voxel volume (binary_fill_holes per slice) vs voxel size",
                "voxel size [m]", "vol_voxel_filled [m³]",
            )

    # ----- alpha-shape sweep, extended, with convex-hull baseline -----
    if not args.skip_alpha:
        rs = _log_space(args.rmax_range[0], args.rmax_range[1], args.n_rmax)
        logger.info("Alpha-radius sweep: %d rmax from %.4f to %.4f m  (n_slices=%d)",
                    args.n_rmax, rs[0], rs[-1], args.n_slices)
        df_a = _sweep_alpha(cluster_points, rs, args.n_slices)
        df_a.to_csv(args.out_dir / "volume_vs_alpha_radius.csv", index=False)

        baseline = _convex_baseline(cluster_points, args.n_slices)
        pd.DataFrame(
            [{"cluster_file": k, "vol_slice_convex": v} for k, v in baseline.items()]
        ).to_csv(args.out_dir / "convex_hull_baseline.csv", index=False)
        for k, v in baseline.items():
            logger.info("  convex-hull baseline %s = %.3f m^3", _short_name(k), v)

        _plot_curves(
            df_a, "rmax", "vol_slice",
            args.out_dir, "volume_vs_alpha_radius",
            f"Slice + alpha-shape volume vs alpha radius (n_slices={args.n_slices})\n"
            f"dashed: per-cluster 2D convex-hull baseline (rmax → ∞ limit)",
            "alpha radius rmax [m]", "vol_slice [m³]",
            horizontal_lines=baseline,
        )


if __name__ == "__main__":
    main()
