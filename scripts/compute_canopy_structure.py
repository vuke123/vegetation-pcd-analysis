#!/usr/bin/env python3
"""
Segment-based canopy structure analysis for vineyard row clusters.

Computes per-row canopy structure proxies from already-segmented,
above-ground point cloud clusters, using LOCAL SEGMENT analysis:

  1. Rotate each row into a local coordinate system (u=along-row, v=cross-row, z=up)
  2. Split each row into small segments along the u axis
  3. Compute canopy metrics (porosity, gap fraction, LAI, LAD) per segment
  4. Aggregate segment metrics into row-level summaries (mean, p50, p90)

WHY SEGMENT-BY-SEGMENT IS NECESSARY:
  The previous approach computed metrics over the full row bounding box.
  Vineyard rows are typically 20-110 m long but only 1-2 m wide.
  A single bounding-box voxelization of a 100 m x 1.5 m x 3 m row creates
  ~4.5 million voxels, but the actual canopy fills only a thin ribbon.
  This inflates porosity to ~0.99, gap fraction to ~0.95, and drives LAI
  proxy down to ~0.1 — values that are physically meaningless because they
  measure how much air surrounds the row, not the canopy's internal structure.
  By computing metrics in small (e.g. 1 m) segments along the row direction,
  each segment's bounding box tightly encloses the local canopy cross-section,
  producing metrics that reflect true canopy openness.

IMPORTANT SCIENTIFIC CAVEATS:
  All metrics are structural proxies derived from discrete-return point clouds
  of already segmented, above-ground vegetation. They are NOT equivalent to
  field-measured LAI or physically modelled LAD. Heights are relative to the
  lowest points in each cluster, not to a true ground surface. These proxies
  are suitable for relative comparison across vine rows within the same flight,
  not for absolute biophysical estimation.

Usage:
    python3 compute_canopy_structure.py --in-dir out_cluster_las
    python3 compute_canopy_structure.py --in-dir out_cluster_las --segment-length 0.5
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
DEFAULT_VOXEL_SIZE = 0.10

# Default segment length along the row direction in metres.
DEFAULT_SEGMENT_LENGTH = 1.0

# Minimum number of points to process a cluster.
MIN_POINTS = 50

# Minimum number of points in a segment for it to be considered valid.
MIN_POINTS_PER_SEGMENT = 10

# G-function value for Beer-Lambert LAI inversion.
# G = 0.5 assumes a spherical leaf angle distribution observed from nadir.
G_FUNCTION = 0.5

# Small epsilon to prevent log(0) in LAI calculation.
EPS = 1e-6

# Minimum number of vertical layers with data to compute LAD profile.
MIN_LAYERS_FOR_LAD = 3


# ---------------------------------------------------------------------------
# 1. Row direction estimation and local coordinate transform
# ---------------------------------------------------------------------------

def _estimate_row_direction(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate row direction from PCA on XY coordinates.

    Returns
    -------
    u_dir : (2,) unit vector along the row (principal axis)
    v_dir : (2,) unit vector across the row (perpendicular)
    """
    centered = xy - xy.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue
    u_dir = eigvecs[:, np.argmax(eigvals)]
    v_dir = np.array([-u_dir[1], u_dir[0]])
    return u_dir, v_dir


def _project_to_local(points: np.ndarray,
                      u_dir: np.ndarray,
                      v_dir: np.ndarray) -> np.ndarray:
    """
    Project 3D points into local (u, v, z) coordinates.

    u = along-row, v = cross-row, z = vertical (unchanged).
    The origin is the XY centroid of the point cloud; z is kept absolute.

    Returns
    -------
    local : (N, 3) array with columns [u, v, z]
    """
    xy = points[:, :2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    u = centered @ u_dir
    v = centered @ v_dir
    z = points[:, 2]
    return np.column_stack([u, v, z])


# ---------------------------------------------------------------------------
# 2. Segmentation along the row axis
# ---------------------------------------------------------------------------

def _split_into_segments(local_points: np.ndarray,
                         segment_length: float
                         ) -> list[np.ndarray]:
    """
    Split points into segments along the u (along-row) axis.

    Each segment covers a window of `segment_length` metres along u.
    Returns a list of (M_i, 3) arrays, one per segment.
    """
    u = local_points[:, 0]
    u_min, u_max = u.min(), u.max()

    segments = []
    start = u_min
    while start < u_max:
        end = start + segment_length
        mask = (u >= start) & (u < end)
        seg_pts = local_points[mask]
        if len(seg_pts) > 0:
            segments.append(seg_pts)
        start = end

    return segments


# ---------------------------------------------------------------------------
# 3. Per-segment metric computation
# ---------------------------------------------------------------------------

def _voxelize_segment(points: np.ndarray,
                      voxel_size: float
                      ) -> tuple[np.ndarray, int, int, int]:
    """
    Voxelize a segment's points and return unique voxels + grid extents.

    The ROI for each segment is its own local bounding box — NOT the full
    row bounding box. This ensures that the voxel grid tightly wraps the
    canopy cross-section in this segment, so porosity and gap fraction
    reflect actual canopy openness, not surrounding air.

    Returns
    -------
    unique_voxels : (M, 3) int array
    nv, nw, nz : grid extents along v, z axes (and u, though typically ~1 voxel wide)
    """
    min_bound = points.min(axis=0)
    indices = np.floor((points - min_bound) / voxel_size).astype(np.int64)
    unique_voxels = np.unique(indices, axis=0)

    bounds = points.max(axis=0) - points.min(axis=0)
    nu = max(int(np.ceil(bounds[0] / voxel_size)), 1)
    nv = max(int(np.ceil(bounds[1] / voxel_size)), 1)
    nz = max(int(np.ceil(bounds[2] / voxel_size)), 1)

    return unique_voxels, nu, nv, nz


def _compute_segment_metrics(points: np.ndarray,
                             voxel_size: float) -> dict | None:
    """
    Compute canopy structure metrics for a single segment.

    All metrics are computed within the segment's local bounding box only.
    This is the core of the segment-by-segment approach.

    Metrics computed:
      - porosity: fraction of empty voxels in the segment's 3D bounding box.
        Unlike the old full-row porosity, this measures canopy openness in a
        tight cross-section, not the emptiness of a 100m-long box.

      - gap_fraction: canopy-wall gap fraction. We project onto the v-z plane
        (the cross-row vertical plane) because vineyard rows are viewed from
        the side for canopy wall assessment. An occupied v-z cell means at
        least one voxel exists at that (v, z) position. Gap fraction is the
        ratio of empty cells to total cells in this cross-section.
        This is more physically meaningful for vineyard rows than top-down
        gap fraction, which for narrow rows is dominated by inter-row space.

      - lai_proxy: Beer-Lambert inversion of the gap fraction.

      - lad_proxy: mean LAD from vertical profile within the segment.
        For each z-layer, gap fraction is computed across the v axis, then
        LAD_layer = -ln(max(gap, eps)) / (G * dz). We report the mean
        across non-empty layers.
    """
    if len(points) < MIN_POINTS_PER_SEGMENT:
        return None

    unique_voxels, nu, nv, nz = _voxelize_segment(points, voxel_size)
    n_total = nu * nv * nz

    if n_total == 0:
        return None

    n_filled = len(unique_voxels)

    # --- Porosity ---
    # Porosity within this segment's tight bounding box.
    porosity = 1.0 - (n_filled / n_total)

    # --- Gap fraction (canopy-wall cross-section, v-z plane) ---
    # Project onto v-z plane: a (v, z) cell is occupied if any u-voxel exists there.
    # For vineyard rows, the canopy wall cross-section is the relevant view.
    vz_columns = set(map(tuple, unique_voxels[:, 1:3]))  # (v_idx, z_idx)
    n_total_vz = nv * nz
    n_occupied_vz = len(vz_columns)
    gap_fraction = 1.0 - (n_occupied_vz / n_total_vz) if n_total_vz > 0 else float("nan")

    # --- LAI proxy (Beer-Lambert inversion of gap fraction) ---
    p_gap = max(gap_fraction, EPS)
    lai_proxy = -np.log(p_gap) / G_FUNCTION

    # --- LAD proxy (vertical profile within segment) ---
    # For each z-layer, compute gap fraction across v, then invert.
    dz = voxel_size
    lad_values = []
    for k in range(nz):
        layer_voxels = unique_voxels[unique_voxels[:, 2] == k]
        if len(layer_voxels) == 0:
            continue  # skip empty layers (above/below canopy)
        layer_v = set(layer_voxels[:, 1].tolist())
        p_gap_layer = 1.0 - (len(layer_v) / nv) if nv > 0 else 1.0
        if p_gap_layer >= 1.0 - EPS:
            continue  # skip near-empty layers
        lad_layer = -np.log(max(p_gap_layer, EPS)) / (G_FUNCTION * dz)
        lad_values.append(lad_layer)

    lad_mean = float(np.mean(lad_values)) if len(lad_values) >= MIN_LAYERS_FOR_LAD else float("nan")

    return {
        "porosity": float(porosity),
        "gap_fraction": float(gap_fraction),
        "lai_proxy": float(lai_proxy),
        "lad_proxy": float(lad_mean),
    }


# ---------------------------------------------------------------------------
# 4. Aggregate segment metrics into row-level summaries
# ---------------------------------------------------------------------------

def _aggregate_segments(segment_results: list[dict],
                        n_total_segments: int) -> dict:
    """
    Aggregate per-segment metrics into row-level mean, p50, p90.

    Only valid (non-None, non-NaN) segments contribute to each statistic.
    """
    n_valid = len(segment_results)

    out = {
        "n_segments_total": n_total_segments,
        "n_segments_valid": n_valid,
    }

    if n_valid == 0:
        for metric in ("porosity", "gap_fraction", "lai_proxy", "lad_proxy"):
            out[f"{metric}_mean"] = float("nan")
            out[f"{metric}_p50"] = float("nan")
            out[f"{metric}_p90"] = float("nan")
        return out

    for metric in ("porosity", "gap_fraction", "lai_proxy", "lad_proxy"):
        values = np.array([s[metric] for s in segment_results
                           if not np.isnan(s[metric])])
        if len(values) == 0:
            out[f"{metric}_mean"] = float("nan")
            out[f"{metric}_p50"] = float("nan")
            out[f"{metric}_p90"] = float("nan")
        else:
            out[f"{metric}_mean"] = float(np.mean(values))
            out[f"{metric}_p50"] = float(np.median(values))
            out[f"{metric}_p90"] = float(np.percentile(values, 90))

    return out


# ---------------------------------------------------------------------------
# 5. CRS extraction
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
# 6. Helpers
# ---------------------------------------------------------------------------

def _parse_row_id(filename: str) -> int | None:
    """Extract cluster/row number from filenames like cluster_03_ndvi.las."""
    m = re.search(r"cluster_(\d+)", filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 7. Per-cluster processing (segment-based)
# ---------------------------------------------------------------------------

def process_cluster(las_path: Path,
                    voxel_size: float,
                    segment_length: float,
                    crs_wkt: str | None = None) -> dict | None:
    """
    Process one cluster LAS file using segment-by-segment canopy analysis.

    Steps:
      1. Load points, estimate row direction via PCA on XY
      2. Rotate into local (u, v, z) coordinates
      3. Split into segments along u
      4. Compute metrics per segment
      5. Aggregate into row-level summaries
    """
    las = laspy.read(str(las_path))
    points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    n_points = len(points)

    if n_points < MIN_POINTS:
        logger.warning("SKIP %s: only %d points (min %d)", las_path.name, n_points, MIN_POINTS)
        return None

    row_id = _parse_row_id(las_path.stem)

    # Step 1: Estimate row direction from PCA on XY
    u_dir, v_dir = _estimate_row_direction(points[:, :2])

    # Step 2: Project into local (u, v, z) coordinate system
    local = _project_to_local(points, u_dir, v_dir)

    # Step 3: Split into segments along u (row direction)
    segments = _split_into_segments(local, segment_length)
    n_total_segments = len(segments)

    # Step 4: Compute metrics per segment
    segment_results = []
    for seg_pts in segments:
        result = _compute_segment_metrics(seg_pts, voxel_size)
        if result is not None:
            segment_results.append(result)

    # Step 5: Aggregate
    agg = _aggregate_segments(segment_results, n_total_segments)

    # Bounding box geometry (original coordinates)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = maxs - mins

    row = {
        # Identity
        "cluster_file": las_path.name,
        "row_id": row_id,
        "point_count": n_points,

        # Segment/voxel params
        "segment_length": segment_length,
        "voxel_size": voxel_size,

        # Aggregated segment metrics
        **agg,

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


# Default voxel sizes used when --voxel-sizes is not supplied but
# --voxel-size is also not supplied and the caller wants multi-size mode.
DEFAULT_VOXEL_SIZES = [0.05, 0.10, 0.15]


# ---------------------------------------------------------------------------
# 8. Helpers for multi-voxel-size mode
# ---------------------------------------------------------------------------

def _voxel_size_tag(voxel_size: float) -> str:
    """
    Convert a voxel size float to a safe filename tag.

    Examples: 0.05 -> 'voxel005', 0.10 -> 'voxel010', 0.15 -> 'voxel015'
    """
    # Round to 3 decimal places to avoid floating-point surprises, then
    # express as an integer number of millimetres (e.g. 0.05 m = 50 mm -> '005').
    mm = round(voxel_size * 1000)
    return f"voxel{mm:03d}"


def _output_path_for_voxel(in_dir: Path,
                           explicit_out: str | None,
                           voxel_size: float,
                           multi_mode: bool) -> Path:
    """
    Resolve the output parquet path for one voxel-size run.

    Single-voxel mode  (multi_mode=False):
      Behaviour is identical to the old code: use --out if given, otherwise
      <in-dir>/row_canopy_structure.parquet.

    Multi-voxel mode (multi_mode=True):
      If --out was given, treat it as a directory and place the per-voxel file
      there; otherwise use <in-dir>.  File name is always
      <tag>_row_canopy_structure.parquet.
    """
    tag = _voxel_size_tag(voxel_size)
    if not multi_mode:
        return Path(explicit_out) if explicit_out else in_dir / "row_canopy_structure.parquet"
    base_dir = Path(explicit_out) if explicit_out else in_dir
    return base_dir / f"{tag}_row_canopy_structure.parquet"


def _run_one_voxel_size(cluster_files: list[Path],
                        voxel_size: float,
                        segment_length: float,
                        out_path: Path,
                        crs_wkt: str | None) -> pd.DataFrame | None:
    """
    Run the full segment-based canopy pipeline for a single voxel size and
    write the result to *out_path*.  Returns the DataFrame on success.
    """
    logger.info("--- voxel_size=%.3f m  ->  %s ---", voxel_size, out_path.name)

    rows = []
    for i, fp in enumerate(cluster_files):
        logger.info("  [%d/%d] %s", i + 1, len(cluster_files), fp.name)
        result = process_cluster(fp, voxel_size, segment_length, crs_wkt)
        if result is not None:
            rows.append(result)

    if not rows:
        logger.error("No rows produced for voxel_size=%.3f.", voxel_size)
        return None

    df = pd.DataFrame(rows)
    if "row_id" in df.columns and df["row_id"].notna().all():
        df = df.sort_values("row_id").reset_index(drop=True)

    df.to_parquet(str(out_path), index=False)
    logger.info("Wrote %d rows to %s", len(df), out_path)
    return df


def _print_summary(df: pd.DataFrame) -> None:
    summary_cols = [
        "row_id", "point_count",
        "n_segments_total", "n_segments_valid",
        "porosity_mean", "porosity_p50",
        "gap_fraction_mean", "gap_fraction_p50",
        "lai_proxy_mean", "lai_proxy_p50",
        "lad_proxy_mean", "lad_proxy_p50",
        "height_range",
    ]
    existing = [c for c in summary_cols if c in df.columns]
    print(f"\nColumns: {list(df.columns)}\n")
    print(df[existing].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Segment-based canopy structure metrics for vineyard row clusters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single voxel size (original behaviour):
  python3 compute_canopy_structure.py --in-dir out_cluster_las
  python3 compute_canopy_structure.py --in-dir out_cluster_las --voxel-size 0.05
  python3 compute_canopy_structure.py --in-dir out_cluster_las --out custom_output.parquet

  # Multiple voxel sizes in one run (separate output file per size):
  python3 compute_canopy_structure.py --in-dir out_cluster_las --voxel-sizes 0.05 0.10 0.15
  python3 compute_canopy_structure.py --in-dir out_cluster_las --voxel-sizes 0.05 0.10 0.15 --out results/

  # Other options:
  python3 compute_canopy_structure.py --in-dir out_cluster_las --segment-length 0.5
  python3 compute_canopy_structure.py --in-dir out_cluster_las --source-las ../datasource/input.las
        """,
    )
    parser.add_argument("--in-dir", required=True,
                        help="Directory containing per-row cluster LAS files")
    parser.add_argument("--out", default=None,
                        help=(
                            "Single-voxel mode: output parquet path "
                            "(default: <in-dir>/row_canopy_structure.parquet). "
                            "Multi-voxel mode: output directory "
                            "(default: <in-dir>); files are named "
                            "<voxelXXX>_row_canopy_structure.parquet."
                        ))
    parser.add_argument("--pattern", default="*_ndvi.las",
                        help="Glob pattern for cluster files (default: *_ndvi.las)")
    parser.add_argument("--voxel-size", type=float, default=None,
                        help=(
                            f"Single voxel edge length in metres "
                            f"(default: {DEFAULT_VOXEL_SIZE}). "
                            "Ignored when --voxel-sizes is provided."
                        ))
    parser.add_argument("--voxel-sizes", type=float, nargs="+", default=None,
                        metavar="VS",
                        help=(
                            "Run the pipeline once per voxel size and write a "
                            "separate parquet for each. "
                            f"Default when this flag is used without values: "
                            f"{DEFAULT_VOXEL_SIZES}. "
                            "Example: --voxel-sizes 0.05 0.10 0.15"
                        ))
    parser.add_argument("--segment-length", type=float, default=DEFAULT_SEGMENT_LENGTH,
                        help=f"Segment length along row direction in metres (default: {DEFAULT_SEGMENT_LENGTH})")
    parser.add_argument("--source-las", default=None,
                        help="Source LAS to extract CRS WKT from")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --- Resolve which voxel sizes to run ---
    multi_mode = args.voxel_sizes is not None
    if multi_mode:
        voxel_sizes = args.voxel_sizes if args.voxel_sizes else DEFAULT_VOXEL_SIZES
    else:
        voxel_sizes = [args.voxel_size if args.voxel_size is not None else DEFAULT_VOXEL_SIZE]

    for vs in voxel_sizes:
        if vs <= 0:
            logger.error("Voxel size must be positive, got %s", vs)
            sys.exit(1)

    if args.segment_length <= 0:
        logger.error("Segment length must be positive, got %s", args.segment_length)
        sys.exit(1)

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        logger.error("%s is not a directory", in_dir)
        sys.exit(1)

    # In multi-voxel mode, --out is treated as a directory; create it if needed.
    if multi_mode and args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    cluster_files = sorted(in_dir.glob(args.pattern))
    if not cluster_files:
        logger.error("No files matching '%s' in %s", args.pattern, in_dir)
        sys.exit(1)

    # CRS (extracted once; shared across all voxel-size runs)
    crs_wkt = None
    if args.source_las:
        crs_wkt = _extract_crs_wkt(Path(args.source_las))
        if crs_wkt:
            logger.info("CRS extracted from %s (%d chars)", args.source_las, len(crs_wkt))
        else:
            logger.warning("No CRS found in %s", args.source_las)

    logger.info(
        "Processing %d cluster files from %s  |  segment=%.2f m  |  voxel sizes: %s",
        len(cluster_files), in_dir, args.segment_length,
        ", ".join(f"{vs:.3f} m" for vs in voxel_sizes),
    )

    # --- Main loop: one full pipeline run per voxel size ---
    produced = []
    for vs in voxel_sizes:
        out_path = _output_path_for_voxel(in_dir, args.out, vs, multi_mode)
        df = _run_one_voxel_size(cluster_files, vs, args.segment_length, out_path, crs_wkt)
        if df is not None:
            produced.append((vs, out_path, df))

    if not produced:
        logger.error("No output produced for any voxel size.")
        sys.exit(1)

    # --- Per-run summaries ---
    for vs, out_path, df in produced:
        print(f"\n{'='*60}")
        print(f"  voxel_size={vs:.3f} m  ->  {out_path.name}")
        print(f"{'='*60}")
        _print_summary(df)


if __name__ == "__main__":
    main()
