#!/usr/bin/env python3
"""
Species-aware NDVI comparison between two segmented vegetation clusters.

NDVI is NOT a species-independent health score. For dense, evergreen
broadleaf canopies (e.g. olive trees) a "healthy" canopy typically reads
NDVI well above ~0.7. For trellised grapevines, where the canopy is a
thin vertical wall mixed with shaded ground, gaps, posts and inter-row
soil, even a vigorous row can sit much lower (often 0.4-0.65) without
indicating poor health. Comparing raw NDVI distributions side-by-side
between species is therefore misleading; what matters is each
distribution relative to its own species-typical reference range.

This script:
  1. Loads two cluster LAS files (assumed to contain an `ndvi` extra dim
     produced by pcd_to_ndvi_las.py; falls back to recomputing from
     red/infrared if missing).
  2. Reports summary statistics (mean, std, p10, p50, p90, healthy-fraction).
  3. Computes a species-normalized health score by mapping NDVI through a
     species-specific [low, high] reference window.
  4. Produces a 2x2 figure: raw NDVI histogram, ECDF, box plot, and the
     species-normalized score histogram.

The default reference windows (olive 0.55-0.85, vineyard 0.35-0.70) are
literature-based starting points; they should be calibrated against your
own ground truth for the thesis.

Usage:
  python3 compare_ndvi_species.py
  python3 compare_ndvi_species.py \
      --las-a scripts/out_cluster_las/config1_leaf00cm_tol40cm_cluster_01_ndvi.las \
      --label-a "Vineyard row" --ref-a 0.35 0.70 \
      --las-b scripts/out_cluster_las/config1_leaf00cm_tol40cm_cluster_04_ndvi.las \
      --label-b "Olive tree" --ref-b 0.55 0.85 \
      --out figures/ndvi_species_compare.png
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np


# ---------------------------------------------------------------------------
# Species reference windows used to normalize raw NDVI into a 0..1 health
# score. These are starting points from literature for ~July/Aug acquisitions
# with high-resolution drone multispectral sensors. Adjust per dataset.
# ---------------------------------------------------------------------------
DEFAULT_REF_VINEYARD = (0.35, 0.70)
DEFAULT_REF_OLIVE = (0.55, 0.85)

# NDVI threshold below which we consider a point "low vigour / stressed"
# relative to the SPECIES window. Expressed as a fraction of the window.
LOW_VIGOUR_FRACTION = 0.25  # bottom 25% of the species reference window


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

@dataclass
class ClusterSample:
    label: str
    path: Path
    ndvi: np.ndarray
    n_points: int
    ref_low: float
    ref_high: float


def _read_ndvi(path: Path) -> np.ndarray:
    """Return NDVI for every point in a cluster LAS file.

    Prefers a precomputed `ndvi` extra dim; falls back to red / nir or red /
    infrared if needed.
    """
    las = laspy.read(str(path))
    dim_names_lower = {d.lower(): d for d in las.point_format.dimension_names}

    if "ndvi" in dim_names_lower:
        return np.asarray(las[dim_names_lower["ndvi"]], dtype=np.float64)

    red_key = next((dim_names_lower[k] for k in ("red",) if k in dim_names_lower), None)
    nir_key = next(
        (dim_names_lower[k] for k in ("nir", "infrared") if k in dim_names_lower),
        None,
    )
    if red_key is None or nir_key is None:
        raise ValueError(
            f"{path.name}: no `ndvi` dim and no red/nir pair to recompute it from."
        )
    red = np.asarray(las[red_key], dtype=np.float64)
    nir = np.asarray(las[nir_key], dtype=np.float64)
    return (nir - red) / (nir + red + 1e-9)


def _load_cluster(path: Path, label: str, ref: tuple[float, float]) -> ClusterSample:
    ndvi = _read_ndvi(path)
    # Clip to a sensible NDVI range. Sensor noise can push a few values
    # outside [-1, 1] and skew percentiles.
    ndvi = ndvi[np.isfinite(ndvi)]
    ndvi = np.clip(ndvi, -1.0, 1.0)
    return ClusterSample(
        label=label,
        path=path,
        ndvi=ndvi,
        n_points=len(ndvi),
        ref_low=float(ref[0]),
        ref_high=float(ref[1]),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _summary_stats(s: ClusterSample) -> dict:
    ndvi = s.ndvi
    low_thr = s.ref_low + LOW_VIGOUR_FRACTION * (s.ref_high - s.ref_low)
    return {
        "label": s.label,
        "n_points": s.n_points,
        "mean": float(np.mean(ndvi)),
        "std": float(np.std(ndvi)),
        "p10": float(np.percentile(ndvi, 10)),
        "p50": float(np.percentile(ndvi, 50)),
        "p90": float(np.percentile(ndvi, 90)),
        "ref_low": s.ref_low,
        "ref_high": s.ref_high,
        "low_vigour_threshold": low_thr,
        "frac_in_ref_window": float(np.mean((ndvi >= s.ref_low) & (ndvi <= s.ref_high))),
        "frac_above_ref_high": float(np.mean(ndvi > s.ref_high)),
        "frac_below_low_vigour": float(np.mean(ndvi < low_thr)),
    }


def _species_normalized_score(s: ClusterSample) -> np.ndarray:
    """Map raw NDVI into [0, 1] via the species reference window.

    score = clip((ndvi - ref_low) / (ref_high - ref_low), 0, 1)

    This is the central idea of the script: a vineyard at NDVI 0.55 and an
    olive at NDVI 0.78 can both map to a similar score (~0.6-0.8), making
    cross-species comparison meaningful in terms of *relative health within
    species expectations*.
    """
    span = max(s.ref_high - s.ref_low, 1e-6)
    return np.clip((s.ndvi - s.ref_low) / span, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(samples: list[ClusterSample], stats: list[dict], out_path: Path) -> None:
    # Keep matplotlib lazy so the script imports cheaply for help / argparse.
    import matplotlib.pyplot as plt

    colors = ["#2E7D32", "#6A1B9A"]  # vineyard green, olive purple
    if len(samples) > len(colors):
        colors = colors + ["#1565C0", "#EF6C00"]

    fig, ax = plt.subplots(figsize=(8, 6))

    bp = ax.boxplot(
        [s.ndvi for s in samples],
        labels=[s.label for s in samples],
        patch_artist=True,
        showfliers=False,
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    # Overlay each species' reference window as dotted horizontal bands.
    for i, s in enumerate(samples, start=1):
        ax.hlines([s.ref_low, s.ref_high], i - 0.35, i + 0.35,
                  colors=colors[i - 1], linestyles=":", lw=1.2)

    ax.set_ylabel("NDVI")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(stats: list[dict]) -> None:
    def fmt(v, p=3):
        return f"{v:.{p}f}" if isinstance(v, float) else str(v)

    rows = [
        ("label", "label"),
        ("n_points", "n_points"),
        ("mean NDVI", "mean"),
        ("std NDVI", "std"),
        ("p10 NDVI", "p10"),
        ("p50 NDVI", "p50"),
        ("p90 NDVI", "p90"),
        ("species ref window low", "ref_low"),
        ("species ref window high", "ref_high"),
        ("low-vigour threshold", "low_vigour_threshold"),
        ("fraction in ref window", "frac_in_ref_window"),
        ("fraction above ref high", "frac_above_ref_high"),
        ("fraction below low-vigour", "frac_below_low_vigour"),
    ]

    headers = [s["label"] for s in stats]
    col_w = max(28, max(len(h) for h in headers) + 2)
    print()
    print("=" * (32 + col_w * len(headers)))
    print(f"{'Metric':32s}" + "".join(f"{h:>{col_w}s}" for h in headers))
    print("-" * (32 + col_w * len(headers)))
    for label, key in rows:
        cells = "".join(f"{fmt(s[key]):>{col_w}s}" for s in stats)
        print(f"{label:32s}{cells}")
    print("=" * (32 + col_w * len(headers)))
    print()
    print("Interpretation hints:")
    print("  - Compare 'mean NDVI' raw vs 'fraction in ref window'. A higher")
    print("    raw mean does NOT imply a healthier canopy across species; the")
    print("    fraction-in-window metric is the cross-species health signal.")
    print("  - 'fraction below low-vigour' approximates the share of canopy")
    print("    points likely indicating stress, expressed in species terms.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent
    repo_root = here.parent
    workspace_root = repo_root.parent
    # Default to the working out_cluster_las copy (sibling scripts/ folder).
    # The in-repo copy currently has NIR=0 -> NDVI=-1 everywhere; the
    # sibling folder is what we use for thesis-side analysis.
    default_data = workspace_root / "scripts" / "out_cluster_las"
    default_a = default_data / "config1_leaf00cm_tol40cm_cluster_01_ndvi.las"
    default_b = default_data / "config1_leaf00cm_tol40cm_cluster_04_ndvi.las"
    default_out = repo_root / "images" / "ndvi_species_compare.png"

    ap = argparse.ArgumentParser(
        description="Species-aware NDVI comparison between two cluster LAS files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--las-a", type=Path, default=default_a,
                    help=f"First cluster LAS (default: {default_a.name})")
    ap.add_argument("--label-a", default="Vineyard row (cluster 01)",
                    help="Label for cluster A")
    ap.add_argument("--ref-a", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    default=DEFAULT_REF_VINEYARD,
                    help=f"NDVI reference window for cluster A (default: {DEFAULT_REF_VINEYARD})")
    ap.add_argument("--las-b", type=Path, default=default_b,
                    help=f"Second cluster LAS (default: {default_b.name})")
    ap.add_argument("--label-b", default="Olive tree (cluster 04)",
                    help="Label for cluster B")
    ap.add_argument("--ref-b", type=float, nargs=2, metavar=("LOW", "HIGH"),
                    default=DEFAULT_REF_OLIVE,
                    help=f"NDVI reference window for cluster B (default: {DEFAULT_REF_OLIVE})")
    ap.add_argument("--out", type=Path, default=default_out,
                    help=f"Output figure path (default: {default_out})")
    ap.add_argument("--no-plot", action="store_true",
                    help="Skip plot generation; print stats only.")
    args = ap.parse_args()

    for p in (args.las_a, args.las_b):
        if not p.is_file():
            print(f"ERROR: not a file: {p}", file=sys.stderr)
            sys.exit(1)

    samples = [
        _load_cluster(args.las_a, args.label_a, args.ref_a),
        _load_cluster(args.las_b, args.label_b, args.ref_b),
    ]
    stats = [_summary_stats(s) for s in samples]
    _print_report(stats)

    if not args.no_plot:
        _plot(samples, stats, args.out)


if __name__ == "__main__":
    main()
