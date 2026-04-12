#!/usr/bin/env python3
"""
Voxel-based canopy structure analysis for vineyard row clusters.

Computes per-row canopy structure proxies from already-segmented,
above-ground point cloud clusters:

  - Voxel porosity (fraction of empty voxels within the canopy bounding box)
  - Gap fraction (top-down and vertical profile)
  - Effective LAI proxy (Beer-Lambert inversion of gap fraction)
  - LAD proxy (layer-wise leaf area density from gap-fraction profile)

IMPORTANT SCIENTIFIC CAVEATS:
  All metrics are structural proxies derived from discrete-return point clouds
  of already segmented, above-ground vegetation. They are NOT equivalent to
  field-measured LAI or physically modelled LAD. Heights are relative to the
  lowest points in each cluster, not to a true ground surface. These proxies
  are suitable for relative comparison across vine rows within the same flight,
  not for absolute biophysical estimation.

Voxelization strategy:
  Uses numpy floor-division voxelization with a fixed default voxel size of
  0.10 m. This was chosen over the notebook's Open3D-based approaches because:
    - 0.10 m gives ~40 vertical layers for typical ~4 m canopy height,
      sufficient for meaningful LAD profiles
    - A fixed size enables fair cross-row comparison (unlike adaptive spacing)
    - Pure numpy: fast, no Open3D dependency, consistent with the pipeline

Usage:
    python3 compute_canopy_structure.py --in-dir out_cluster_las
    python3 compute_canopy_structure.py --in-dir out_cluster_las --voxel-size 0.05
    python3 compute_canopy_structure.py --in-dir out_cluster_las --out results.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default voxel edge length in metres.
# 0.10 m balances resolution (captures gaps) with stability (enough points
# per voxel at typical multispectral drone point densities of ~50-200 pts/m²).
DEFAULT_VOXEL_SIZE = 0.10

# Minimum number of points to process a cluster.
MIN_POINTS = 50

# G-function value for Beer-Lambert LAI inversion.
# G = 0.5 assumes a spherical leaf angle distribution observed from nadir —
# a standard first-order assumption when no LIDF data is available.
G_FUNCTION = 0.5

# Small epsilon to prevent log(0) in LAI calculation.
EPS = 1e-6

# Minimum number of vertical layers with data to compute LAD profile.
MIN_LAYERS_FOR_LAD = 3


# ---------------------------------------------------------------------------
# 1. Voxelization
# ---------------------------------------------------------------------------

def voxelize(points: np.ndarray,
             voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each point to a 3D voxel via floor division.

    Returns
    -------
    indices : (N, 3) int array — voxel grid indices per point
    unique_voxels : (M, 3) int array — unique occupied voxel indices
    """
    min_bound = points.min(axis=0)
    indices = np.floor((points - min_bound) / voxel_size).astype(np.int64)
    unique_voxels = np.unique(indices, axis=0)
    return indices, unique_voxels


def _grid_extent(points: np.ndarray,
                 voxel_size: float) -> tuple[int, int, int]:
    """Number of voxels along each axis for the bounding box."""
    bounds = points.max(axis=0) - points.min(axis=0)
    nx = max(int(np.ceil(bounds[0] / voxel_size)), 1)
    ny = max(int(np.ceil(bounds[1] / voxel_size)), 1)
    nz = max(int(np.ceil(bounds[2] / voxel_size)), 1)
    return nx, ny, nz


# ---------------------------------------------------------------------------
# 2. Voxel porosity
# ---------------------------------------------------------------------------

def compute_voxel_porosity(points: np.ndarray,
                           voxel_size: float) -> dict:
    """
    Porosity = 1 - (filled voxels / total voxels in bounding box).

    The bounding box of the cluster defines the canopy region-of-interest.
    For vine rows this is reasonable because the clustering step has already
    isolated each row, so the bbox tightly encloses the canopy.
    """
    _, unique = voxelize(points, voxel_size)
    nx, ny, nz = _grid_extent(points, voxel_size)
    n_total = nx * ny * nz
    n_filled = len(unique)
    fill_ratio = n_filled / n_total if n_total > 0 else float("nan")
    porosity = 1.0 - fill_ratio

    return {
        "voxel_size": voxel_size,
        "n_voxels_total": n_total,
        "n_voxels_filled": n_filled,
        "voxel_fill_ratio": float(fill_ratio),
        "voxel_porosity": float(porosity),
        "grid_nx": nx,
        "grid_ny": ny,
        "grid_nz": nz,
    }


# ---------------------------------------------------------------------------
# 3. Gap fraction
# ---------------------------------------------------------------------------

def compute_gap_fraction(points: np.ndarray,
                         voxel_size: float) -> dict:
    """
    Gap fraction computed two ways:

    Top-down (gap_fraction_top):
      Project all filled voxels onto the XY plane. An XY column is "occupied"
      if at least one voxel in that column contains points. Gap fraction is
      the ratio of empty columns to total columns in the bounding-box footprint.

    Vertical profile:
      For each horizontal Z-layer, compute the fraction of empty XY cells
      within the layer's footprint. Report mean, median, and p90 across layers.
    """
    indices, unique = voxelize(points, voxel_size)
    nx, ny, nz = _grid_extent(points, voxel_size)

    # --- Top-down gap fraction ---
    xy_columns = set(map(tuple, unique[:, :2]))
    n_total_columns = nx * ny
    n_occupied_columns = len(xy_columns)
    gap_top = 1.0 - (n_occupied_columns / n_total_columns) if n_total_columns > 0 else float("nan")

    # --- Vertical profile ---
    layer_gaps = []
    for k in range(nz):
        layer_mask = unique[:, 2] == k
        layer_voxels = unique[layer_mask]
        if len(layer_voxels) == 0:
            # Entire layer is empty — gap fraction = 1.0
            layer_gaps.append(1.0)
            continue
        layer_xy = set(map(tuple, layer_voxels[:, :2]))
        # Use the per-layer XY extent (nx * ny) as the denominator.
        # This treats the full bounding-box footprint as the reference area
        # for each layer, which is appropriate for vine rows where the
        # footprint is roughly constant along the height.
        gap = 1.0 - (len(layer_xy) / n_total_columns) if n_total_columns > 0 else float("nan")
        layer_gaps.append(gap)

    layer_gaps = np.array(layer_gaps)

    return {
        "gap_fraction_top": float(gap_top),
        "gap_fraction_profile_mean": float(np.mean(layer_gaps)),
        "gap_fraction_profile_p50": float(np.median(layer_gaps)),
        "gap_fraction_profile_p90": float(np.percentile(layer_gaps, 90)),
        "n_vertical_layers": nz,
    }


# ---------------------------------------------------------------------------
# 4. LAI proxy (effective)
# ---------------------------------------------------------------------------

def compute_lai_proxy(gap_fraction_top: float) -> dict:
    """
    Effective LAI proxy via Beer-Lambert inversion of top-down gap fraction.

        LAI_eff = -ln(max(P_gap, eps)) / G

    where G = 0.5 (spherical LIDF, nadir-like viewing).

    This is a PROXY: the true Beer-Lambert model assumes randomly distributed
    foliage and a known G-function. For structured vine canopies this
    overestimates clumped LAI but is useful for relative comparison.
    """
    p_gap = max(gap_fraction_top, EPS)
    lai = -np.log(p_gap) / G_FUNCTION

    return {
        "lai_proxy_eff": float(lai),
    }


# ---------------------------------------------------------------------------
# 5. LAD proxy (vertical profile)
# ---------------------------------------------------------------------------

def compute_lad_proxy(points: np.ndarray,
                      voxel_size: float) -> dict:
    """
    LAD-like vertical proxy from the gap-fraction profile.

    For each horizontal Z-layer of thickness dz = voxel_size, estimate a
    local extinction coefficient:

        LAD_layer = -ln(max(P_gap_layer, eps)) / (G * dz)

    where P_gap_layer is the gap fraction of that layer (fraction of empty
    XY cells). This is analogous to a voxel-based contact frequency.

    We report summary statistics (mean, median, p90) across non-empty layers
    only. Layers where the gap fraction is exactly 1.0 (completely empty) are
    excluded from statistics because they typically represent space above or
    below the canopy rather than within-canopy gaps.

    CAVEAT: This is NOT a true LAD measurement. It is a structural density
    proxy suitable for relative comparison across vine rows within the same
    acquisition. It does not account for beam geometry, multi-return effects,
    or leaf angle distribution.
    """
    indices, unique = voxelize(points, voxel_size)
    nx, ny, nz = _grid_extent(points, voxel_size)
    n_total_columns = nx * ny

    if n_total_columns == 0 or nz < MIN_LAYERS_FOR_LAD:
        return {
            "lad_proxy_mean": float("nan"),
            "lad_proxy_p50": float("nan"),
            "lad_proxy_p90": float("nan"),
        }

    dz = voxel_size
    lad_values = []

    for k in range(nz):
        layer_voxels = unique[unique[:, 2] == k]
        if len(layer_voxels) == 0:
            continue  # skip fully empty layers (above/below canopy)
        layer_xy = set(map(tuple, layer_voxels[:, :2]))
        p_gap = 1.0 - (len(layer_xy) / n_total_columns)
        if p_gap >= 1.0 - EPS:
            continue  # skip layers with near-zero occupancy
        lad_layer = -np.log(max(p_gap, EPS)) / (G_FUNCTION * dz)
        lad_values.append(lad_layer)

    if len(lad_values) < MIN_LAYERS_FOR_LAD:
        return {
            "lad_proxy_mean": float("nan"),
            "lad_proxy_p50": float("nan"),
            "lad_proxy_p90": float("nan"),
        }

    lad_arr = np.array(lad_values)
    return {
        "lad_proxy_mean": float(np.mean(lad_arr)),
        "lad_proxy_p50": float(np.median(lad_arr)),
        "lad_proxy_p90": float(np.percentile(lad_arr, 90)),
    }


# ---------------------------------------------------------------------------
# 6. CRS extraction (reused from compute_row_features.py)
# ---------------------------------------------------------------------------

def _extract_crs_wkt(las_path: Path) -> str | None:
    try:
        with laspy.open(str(las_path)) as f:
            for vlr in f.header.vlrs:
                if hasattr(vlr, "string") and vlr.string:
                    return vlr.string.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 7. Helpers
# ---------------------------------------------------------------------------

def _parse_row_id(filename: str) -> int | None:
    """Extract cluster/row number from filenames like cluster_03_ndvi.las."""
    m = re.search(r"cluster_(\d+)", filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 8. Per-cluster processing
# ---------------------------------------------------------------------------

def process_cluster(las_path: Path,
                    voxel_size: float,
                    crs_wkt: str | None = None) -> dict | None:
    """Process one cluster LAS file and return a dict of canopy structure metrics."""
    las = laspy.read(str(las_path))
    points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    n_points = len(points)

    if n_points < MIN_POINTS:
        logger.warning("SKIP %s: only %d points (min %d)", las_path.name, n_points, MIN_POINTS)
        return None

    row_id = _parse_row_id(las_path.stem)

    # Bounding box geometry
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins

    # Core metrics
    porosity = compute_voxel_porosity(points, voxel_size)
    gap = compute_gap_fraction(points, voxel_size)
    lai = compute_lai_proxy(gap["gap_fraction_top"])
    lad = compute_lad_proxy(points, voxel_size)

    row = {
        # Identity
        "cluster_file": las_path.name,
        "row_id": row_id,
        "point_count": n_points,

        # Voxel structure
        **porosity,

        # Gap fraction
        **gap,

        # LAI proxy
        **lai,

        # LAD proxy
        **lad,

        # Supporting geometry
        "bbox_x_extent": float(extents[0]),
        "bbox_y_extent": float(extents[1]),
        "bbox_z_extent": float(extents[2]),
        "height_range": float(extents[2]),
        "bbox_minz": float(mins[2]),
        "bbox_maxz": float(maxs[2]),
    }

    if crs_wkt:
        row["crs_wkt"] = crs_wkt

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute voxel-based canopy structure metrics for vineyard row clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 compute_canopy_structure.py --in-dir out_cluster_las
  python3 compute_canopy_structure.py --in-dir out_cluster_las --voxel-size 0.05
  python3 compute_canopy_structure.py --in-dir out_cluster_las --out custom_output.parquet
  python3 compute_canopy_structure.py --in-dir out_cluster_las --source-las ../datasource/input.las
        """,
    )
    parser.add_argument("--in-dir", required=True,
                        help="Directory containing per-row cluster LAS files")
    parser.add_argument("--out", default=None,
                        help="Output parquet path (default: <in-dir>/row_canopy_structure.parquet)")
    parser.add_argument("--pattern", default="*_ndvi.las",
                        help="Glob pattern for cluster files (default: *_ndvi.las)")
    parser.add_argument("--voxel-size", type=float, default=DEFAULT_VOXEL_SIZE,
                        help=f"Voxel edge length in metres (default: {DEFAULT_VOXEL_SIZE})")
    parser.add_argument("--source-las", default=None,
                        help="Source LAS to extract CRS WKT from")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        logger.error("%s is not a directory", in_dir)
        sys.exit(1)

    out_path = Path(args.out) if args.out else in_dir / "row_canopy_structure.parquet"

    if args.voxel_size <= 0:
        logger.error("Voxel size must be positive, got %s", args.voxel_size)
        sys.exit(1)

    cluster_files = sorted(in_dir.glob(args.pattern))
    if not cluster_files:
        logger.error("No files matching '%s' in %s", args.pattern, in_dir)
        sys.exit(1)

    # CRS
    crs_wkt = None
    if args.source_las:
        crs_wkt = _extract_crs_wkt(Path(args.source_las))
        if crs_wkt:
            logger.info("CRS extracted from %s (%d chars)", args.source_las, len(crs_wkt))
        else:
            logger.warning("No CRS found in %s", args.source_las)

    logger.info("Processing %d cluster files from %s (voxel_size=%.3f m)",
                len(cluster_files), in_dir, args.voxel_size)

    rows = []
    for i, fp in enumerate(cluster_files):
        logger.info("  [%d/%d] %s", i + 1, len(cluster_files), fp.name)
        result = process_cluster(fp, args.voxel_size, crs_wkt)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("No rows produced.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    if "row_id" in df.columns and df["row_id"].notna().all():
        df = df.sort_values("row_id").reset_index(drop=True)

    df.to_parquet(str(out_path), index=False)

    logger.info("Wrote %d rows to %s", len(df), out_path)

    # Summary
    summary_cols = [
        "row_id", "point_count",
        "voxel_porosity", "voxel_fill_ratio",
        "gap_fraction_top", "gap_fraction_profile_mean",
        "lai_proxy_eff",
        "lad_proxy_mean", "lad_proxy_p50",
        "height_range",
    ]
    existing = [c for c in summary_cols if c in df.columns]
    print(f"\nColumns: {list(df.columns)}\n")
    print(df[existing].to_string(index=False))


if __name__ == "__main__":
    main()
