#!/usr/bin/env python3
"""
pipeline_snapshots.py — render one PNG per pipeline checkpoint.

Purpose
-------
Give a visual "before/after" of how a single drone point cloud is transformed as
it moves through run_pipeline.sh, so the difference between each stage is obvious
at a glance. This script only *reads* the on-disk outputs that the pipeline has
already produced — it never re-runs or modifies anything.

Checkpoints produced (top-down / X-Y views for spatial consistency):
  1. raw_pointcloud.png   - full raw drone capture, true-colour (RGB) orthophoto,
                            with the processed ROI outlined.
  2. ground_removed.png   - SMRF classification (ground vs vegetation) next to the
                            non-ground cloud that survives into clustering.
  3. clustered_rows.png   - the non-ground cloud split into vine rows, each row a
                            distinct colour (one colour per cluster LAS).
  4. ndvi_metrics.png     - NDVI raster (recomputed from the raw NIR/Red bands,
                            because the pipeline's clustered PCDs drop NIR) plus a
                            per-row metrics table read from row_features.parquet.
  0. pipeline_overview.png- all four checkpoints stitched into one montage.

Nothing here is on the pipeline's critical path; it's a documentation / thesis aid.
Run it after run_pipeline.sh has completed at least once.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a GUI window
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

import laspy


# --------------------------------------------------------------------------- #
# Low-level LAS helpers
# --------------------------------------------------------------------------- #
def _band(pts, *names):
    """Return the first band present among `names` (case-insensitive) or None."""
    have = {n.lower(): n for n in pts.point_format.dimension_names}
    for n in names:
        if n.lower() in have:
            return np.asarray(pts[have[n.lower()]])
    return None


def read_xy(path, max_points=None):
    """Read x, y (and z) from a small LAS fully, optionally strided to max_points."""
    f = laspy.read(str(path))
    x = np.asarray(f.x, dtype=np.float64)
    y = np.asarray(f.y, dtype=np.float64)
    z = np.asarray(f.z, dtype=np.float64)
    if max_points and len(x) > max_points:
        step = int(np.ceil(len(x) / max_points))
        x, y, z = x[::step], y[::step], z[::step]
    return x, y, z


def bin_bands(path, extent, cell, bands, chunk=2_000_000):
    """
    Stream a (possibly huge) LAS in chunks and accumulate per-cell band *sums* and
    a point *count* over a fixed grid. Returns (grids, count, nx, ny).

    `grids[name]` and `count` are 2-D arrays indexed [row=y, col=x]. Summing in a
    single pass keeps peak memory flat even for the 30M-point raw capture.
    """
    xmin, xmax, ymin, ymax = extent
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))
    grids = {b: np.zeros((ny, nx), dtype=np.float64) for b in bands}
    count = np.zeros((ny, nx), dtype=np.float64)

    with laspy.open(str(path)) as reader:
        for pts in reader.chunk_iterator(chunk):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            m = (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)
            if not m.any():
                continue
            x, y = x[m], y[m]
            ix = ((x - xmin) / cell).astype(np.int64)
            iy = ((y - ymin) / cell).astype(np.int64)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            flat = iy * nx + ix
            count += np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
            for b in bands:
                vals = _band(pts, b)
                if vals is None:
                    continue
                vals = vals[m].astype(np.float64)
                grids[b] += np.bincount(flat, weights=vals, minlength=nx * ny).reshape(ny, nx)
    return grids, count, nx, ny


# --------------------------------------------------------------------------- #
# Checkpoint renderers
# --------------------------------------------------------------------------- #
def _pct_norm(a, lo=2, hi=98):
    """Normalise to [0,1] using robust percentiles (ignores hot/dark outliers)."""
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a)
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    return np.clip((a - p_lo) / (p_hi - p_lo), 0, 1)


def render_raw(raw_las, roi, out_png, cell=0.30):
    """Checkpoint 1 — raw capture as an RGB orthophoto, ROI outlined."""
    f0 = laspy.open(str(raw_las))
    h = f0.header
    extent = (h.mins[0], h.maxs[0], h.mins[1], h.maxs[1])
    grids, count, nx, ny = bin_bands(raw_las, extent, cell, ["red", "green", "blue"])

    have_rgb = count.sum() > 0 and grids["red"].sum() > 0
    fig, ax = plt.subplots(figsize=(9, 9 * ny / max(nx, 1)))
    with np.errstate(invalid="ignore", divide="ignore"):
        if have_rgb:
            rgb = np.dstack([
                _pct_norm(np.where(count > 0, grids[b] / count, np.nan))
                for b in ("red", "green", "blue")
            ])
            rgb = np.nan_to_num(rgb)
            rgb[count == 0] = 1.0  # empty cells -> white, matching the other panels
            ax.imshow(rgb, origin="lower",
                      extent=(extent[0], extent[1], extent[2], extent[3]))
        else:  # no colour bands -> fall back to point-density
            dens = np.where(count > 0, count, np.nan)
            ax.imshow(dens, origin="lower", cmap="Greys_r",
                      extent=(extent[0], extent[1], extent[2], extent[3]))

    xmin, xmax, ymin, ymax = roi
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, edgecolor="red", lw=2.0, ls="--"))
    ax.text(xmin, ymax, " obrađeno područje (ROI)", color="red", va="bottom",
            ha="left", fontsize=10, fontweight="bold")
    ax.set_title("1 · Sirovi oblak točaka iz drona (prave boje)\n"
                 f"{Path(raw_las).name}", fontsize=12)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def render_ground_removed(classified_las, nonground_las, roi, out_png):
    """Checkpoint 2 — the retained non-ground cloud, coloured by height."""
    x, y, z = read_xy(nonground_las, max_points=1_200_000)
    xmin, xmax, ymin, ymax = roi

    fig, ax = plt.subplots(figsize=(8, 8 * (ymax - ymin) / (xmax - xmin)))
    zc = z - np.percentile(z, 1)
    sc = ax.scatter(x, y, s=0.5, c=zc, cmap="viridis",
                    vmin=0, vmax=np.percentile(zc, 99), linewidths=0)
    ax.set_title("2 · Tlo uklonjeno — oblak vegetacije "
                 f"(prikazano {len(x):,} toč.)", fontsize=12)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("visina iznad tla [m]")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def render_clusters(cluster_files, out_png, per_cluster_cap=60_000):
    """Checkpoint 3 — each cluster LAS drawn in its own colour = the vine rows."""
    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = cm.get_cmap("tab20")
    for i, path in enumerate(cluster_files):
        x, y, _ = read_xy(path, max_points=per_cluster_cap)
        ax.scatter(x, y, s=0.6, color=cmap(i % 20), linewidths=0,
                   label=f"red {i:02d}")
    ax.set_title(f"3 · Grupirani redovi trsa ({len(cluster_files)} grupa)",
                 fontsize=12)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ncol = 2 if len(cluster_files) > 10 else 1
    ax.legend(markerscale=10, fontsize=8, ncol=ncol, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def compute_ndvi_raster(raw_las, roi, cell=0.15):
    """NDVI = (NIR-Red)/(NIR+Red) binned to a grid over the ROI, from raw bands."""
    grids, count, nx, ny = bin_bands(raw_las, roi, cell, ["nir", "red"])
    with np.errstate(invalid="ignore", divide="ignore"):
        nir = np.where(count > 0, grids["nir"] / count, np.nan)
        red = np.where(count > 0, grids["red"] / count, np.nan)
        ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi[count == 0] = np.nan
    return ndvi, count


def render_ndvi_metrics(raw_las, roi, parquet, cluster_files, out_png, cell=0.15):
    """Checkpoint 4 — NDVI raster (recomputed from the raw NIR/Red bands)."""
    ndvi, count = compute_ndvi_raster(raw_las, roi, cell)
    xmin, xmax, ymin, ymax = roi

    fig, ax_map = plt.subplots(figsize=(8, 8 * (ymax - ymin) / (xmax - xmin)))
    finite = ndvi[np.isfinite(ndvi)]
    vmin, vmax = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    im = ax_map.imshow(ndvi, origin="lower", cmap="RdYlGn",
                       vmin=vmin, vmax=vmax,
                       extent=(xmin, xmax, ymin, ymax))
    stats = (f"   (polje: sred. {np.nanmean(finite):.3f}, "
             f"p10 {np.nanpercentile(finite, 10):.3f}, "
             f"p90 {np.nanpercentile(finite, 90):.3f})" if finite.size else "")
    ax_map.set_title(f"4 · NDVI raster (iz sirovih NIR/Crveno, ćelije {cell:g} m)\n"
                     + stats.strip(), fontsize=12)
    ax_map.set_xlabel("X [m]"); ax_map.set_ylabel("Y [m]")
    ax_map.set_aspect("equal")
    cb = fig.colorbar(im, ax=ax_map, shrink=0.85)
    cb.set_label("NDVI")

    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def render_overview(pngs, out_png):
    """
    Stitch the four checkpoint PNGs into a clean 2x2 board.

    Each panel keeps its own aspect ratio (imshow never distorts), but every
    panel is scaled to the *same height* and its cell is sized to its width, so
    the two rows read as aligned image strips instead of floating, mismatched
    tiles. Axes are placed manually in figure coordinates to control the gutters.
    """
    import matplotlib.image as mpimg

    labels = ["sirovi podaci", "uklonjeno tlo", "grupirani redovi", "NDVI raster"]
    imgs = [mpimg.imread(p) if (p and Path(p).is_file()) else None for p in pngs]
    # width/height aspect of each image (fallback square if missing)
    aspects = [(im.shape[1] / im.shape[0]) if im is not None else 1.0 for im in imgs]

    # Layout units (inches): common panel height, gutters, and title band.
    H = 6.0                 # every panel is this tall
    gx, gy = 0.5, 1.0       # horizontal gap in a row, vertical gap between rows
    m = 0.4                 # outer margin
    title_h = 0.9           # space reserved at the top for the suptitle
    label_h = 0.35          # space above each row for panel labels

    rows = [(0, 1), (2, 3)]
    widths = [a * H for a in aspects]
    row_w = [widths[i] + gx + widths[j] for (i, j) in rows]
    fig_w = max(row_w) + 2 * m
    fig_h = 2 * H + gy + 2 * label_h + title_h + 2 * m

    fig = plt.figure(figsize=(fig_w, fig_h))
    for r, (i, j) in enumerate(rows):
        # top row sits higher; y measured from the bottom in inches
        row_bottom = m + (1 - r) * (H + gy + label_h)
        total = widths[i] + gx + widths[j]
        x = m + (fig_w - 2 * m - total) / 2.0   # centre the row
        for k in (i, j):
            ax = fig.add_axes([x / fig_w, row_bottom / fig_h,
                               widths[k] / fig_w, H / fig_h])
            ax.axis("off")
            if imgs[k] is not None:
                ax.imshow(imgs[k])
            # panel label centred just above the image
            fig.text((x + widths[k] / 2) / fig_w,
                     (row_bottom + H + 0.12) / fig_h,
                     labels[k], ha="center", va="bottom", fontsize=14)
            x += widths[k] + gx

    fig.suptitle("Cjevovod obrade oblaka točaka vinograda — kontrolne točke po fazama",
                 fontsize=17, y=1 - (m * 0.4) / fig_h)
    fig.savefig(out_png, dpi=110, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #
def find_one(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def main():
    here = Path(__file__).resolve().parent
    scripts_dir = here.parent  # visualization/ -> scripts/

    # The raw cloud's location differs between checkouts (datasource/ can sit
    # beside scripts/ or one level up at the repo root). Auto-discover the file
    # the pipeline was run on by deriving its name from a non_ground output.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-las", default=None,
                    help="raw input cloud fed into run_pipeline.sh "
                         "(auto-detected from datasource/ if omitted)")
    ap.add_argument("--out-ground-dir", default=str(scripts_dir / "out_ground"))
    ap.add_argument("--out-cluster-las-dir",
                    default=str(scripts_dir / "out_cluster_las"))
    ap.add_argument("--out-dir", default=str(scripts_dir / "pipeline_snapshots"),
                    help="folder to write the checkpoint PNGs into")
    ap.add_argument("--roi-margin", type=float, default=1.0,
                    help="metres of padding around the processed extent")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nonground = find_one(os.path.join(args.out_ground_dir, "*_non_ground.las"))

    # Resolve the raw cloud: derive its basename from the non_ground filename
    # (strip the pipeline's "_classified_smrf_non_ground" suffix) and look for it
    # in the usual datasource/ locations relative to the repo.
    raw_las = args.raw_las
    if raw_las is None and nonground:
        stem = Path(nonground).name
        for suf in ("_classified_smrf_non_ground.las", "_non_ground.las"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        candidates = []
        for base in (scripts_dir.parent, scripts_dir.parent.parent,
                     scripts_dir.parent.parent.parent):
            candidates += glob.glob(str(base / "datasource" / "**" / f"{stem}.las"),
                                    recursive=True)
        raw_las = candidates[0] if candidates else None
        if raw_las:
            print(f"Auto-detected raw LAS: {raw_las}")
    if raw_las is None:
        raw_las = ""
    args.raw_las = raw_las

    classified = None
    for c in sorted(glob.glob(os.path.join(args.out_ground_dir, "*_smrf.las"))):
        if "non_ground" not in c and "_ground" not in Path(c).stem[-7:]:
            classified = c
            break
    if classified is None:  # fallback: any *_smrf.las that isn't a split output
        for c in sorted(glob.glob(os.path.join(args.out_ground_dir, "*.las"))):
            s = Path(c).stem
            if s.endswith("_smrf"):
                classified = c
                break

    cluster_files = sorted(
        p for p in glob.glob(os.path.join(args.out_cluster_las_dir, "*_ndvi.las"))
        if "merged" not in Path(p).name
    )
    parquet = os.path.join(args.out_cluster_las_dir, "row_features.parquet")

    if not nonground:
        raise SystemExit(f"No *_non_ground.las in {args.out_ground_dir}; "
                         "run run_pipeline.sh first.")
    if not cluster_files:
        raise SystemExit(f"No *_ndvi.las clusters in {args.out_cluster_las_dir}; "
                         "run run_pipeline.sh first.")

    # ROI = processed (non-ground) extent + margin. All post-raw views share it.
    nx, ny, nz = read_xy(nonground, max_points=400_000)
    m = args.roi_margin
    roi = (nx.min() - m, nx.max() + m, ny.min() - m, ny.max() + m)
    print(f"Processed ROI (X,Y): {roi}")

    p1 = out_dir / "1_raw_pointcloud.png"
    p2 = out_dir / "2_ground_removed.png"
    p3 = out_dir / "3_clustered_rows.png"
    p4 = out_dir / "4_ndvi_metrics.png"
    p0 = out_dir / "0_pipeline_overview.png"

    print("Rendering checkpoints:")
    if Path(args.raw_las).is_file():
        render_raw(args.raw_las, roi, str(p1))
    else:
        print(f"  ! raw LAS not found ({args.raw_las}); skipping checkpoint 1")
        p1 = None

    render_ground_removed(classified, nonground, roi, str(p2))
    render_clusters(cluster_files, str(p3))

    if Path(args.raw_las).is_file():
        render_ndvi_metrics(args.raw_las, roi, parquet, cluster_files, str(p4))
    else:
        print("  ! raw LAS missing; NDVI raster needs raw NIR/Red — skipping 4")
        p4 = None

    render_overview([str(p1) if p1 else None, str(p2), str(p3),
                     str(p4) if p4 else None], str(p0))

    print(f"\nDone. PNGs in: {out_dir}")


if __name__ == "__main__":
    main()
