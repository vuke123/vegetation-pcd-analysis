#!/usr/bin/env python3
"""
Compute per-row features for vineyard cluster LAS files.

For each cluster LAS file, computes:
  - Slope-aware height (RANSAC ground-plane approximation from lowest points)
  - Row geometry: length, width, azimuth via PCA
  - Volume: voxel-based and slice+alpha-hull
  - NDVI statistics: mean, std, p10, p90, low_frac

Outputs: row_features.parquet in the output directory.

Usage:
    python3 compute_row_features.py --in-dir out_cluster_las --out row_features.parquet
    python3 compute_row_features.py --in-dir out_cluster_las  # default: out in same dir
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from shapely.geometry import LineString
from shapely.ops import polygonize


# ---------------------------------------------------------------------------
# 1. Slope-aware height estimation
# ---------------------------------------------------------------------------

def _ransac_ground_plane(points: np.ndarray,
                         low_fraction: float = 0.10,
                         n_iter: int = 200,
                         dist_thresh: float = 0.10) -> tuple[np.ndarray, float]:
    """
    Fit a plane to the lowest `low_fraction` of points by Z using RANSAC.

    Returns (normal, d) where the plane equation is normal . p + d = 0,
    with normal pointing upward (nz > 0).
    """
    z = points[:, 2]
    z_cutoff = np.percentile(z, low_fraction * 100)
    low_pts = points[z <= z_cutoff]

    if len(low_pts) < 3:
        # Fallback: horizontal plane at zmin
        return np.array([0.0, 0.0, 1.0]), -float(z.min())

    best_inliers = 0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = -float(z.min())
    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        idx = rng.choice(len(low_pts), 3, replace=False)
        p0, p1, p2 = low_pts[idx]
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n = n / norm
        if n[2] < 0:
            n = -n
        # Reject near-vertical planes (slope > 45 deg is not ground)
        if n[2] < 0.707:
            continue
        d = -np.dot(n, p0)

        dists = np.abs(low_pts @ n + d)
        inliers = int(np.sum(dists < dist_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = n
            best_d = d

    return best_normal, best_d


def compute_slope_aware_height(points: np.ndarray) -> dict:
    """
    Compute height metrics relative to a RANSAC-fitted ground plane
    from the lowest points in the cluster.

    Returns dict with:
      height_max, height_mean, height_p50, height_p90,
      ground_slope_deg, ground_z_mean
    """
    normal, d = _ransac_ground_plane(points)

    # Signed height of every point above the ground plane
    heights = points @ normal + d

    # The minimum height should be ~0 for ground-adjacent points.
    # Shift so that the 2nd percentile is zero (robust against noise).
    h_floor = np.percentile(heights, 2)
    heights = heights - h_floor

    slope_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))

    z_low = points[:, 2]
    z_cutoff = np.percentile(z_low, 10)
    ground_z_mean = float(np.mean(z_low[z_low <= z_cutoff]))

    h_mean = float(np.mean(heights))
    h_std = float(np.std(heights))
    h_cv = (h_std / h_mean) if h_mean > 0 else float("nan")

    return {
        "height_max": float(np.max(heights)),
        "height_mean": h_mean,
        "height_std": h_std,
        "height_cv": h_cv,
        "height_p50": float(np.median(heights)),
        "height_p90": float(np.percentile(heights, 90)),
        "ground_slope_deg": slope_deg,
        "ground_z_mean": ground_z_mean,
    }


# ---------------------------------------------------------------------------
# 2. Row geometry via PCA
# ---------------------------------------------------------------------------

def compute_row_geometry(points: np.ndarray) -> dict:
    """
    Compute row direction, length, and width using PCA on XY coordinates.

    Returns dict with:
      row_length, row_width, azimuth_deg,
      centroid_x, centroid_y, centroid_z,
      bbox_minx..bbox_maxz
    """
    xy = points[:, :2]
    centroid_xy = xy.mean(axis=0)
    centered = xy - centroid_xy

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Principal axis = eigenvector with largest eigenvalue
    principal = eigvecs[:, np.argmax(eigvals)]

    # Project onto principal axis and its perpendicular
    proj_along = centered @ principal
    perp = np.array([-principal[1], principal[0]])
    proj_perp = centered @ perp

    row_length = float(proj_along.max() - proj_along.min())
    row_width = float(proj_perp.max() - proj_perp.min())

    # Azimuth: angle of principal axis from north (Y+), clockwise
    # principal is (dx, dy); azimuth = atan2(dx, dy) in degrees
    azimuth = float(np.degrees(np.arctan2(principal[0], principal[1]))) % 360

    return {
        "row_length": row_length,
        "row_width": row_width,
        "azimuth_deg": azimuth,
        "centroid_x": float(centroid_xy[0]),
        "centroid_y": float(centroid_xy[1]),
        "centroid_z": float(points[:, 2].mean()),
        "bbox_minx": float(points[:, 0].min()),
        "bbox_miny": float(points[:, 1].min()),
        "bbox_minz": float(points[:, 2].min()),
        "bbox_maxx": float(points[:, 0].max()),
        "bbox_maxy": float(points[:, 1].max()),
        "bbox_maxz": float(points[:, 2].max()),
    }


# ---------------------------------------------------------------------------
# 3a. Voxel-based volume
# ---------------------------------------------------------------------------

def compute_voxel_volume(points: np.ndarray, voxel_size: float | None = None) -> dict:
    """
    Volume = n_voxels * voxel_size^3.
    If voxel_size is None, use 1% of the largest bounding-box extent (same
    heuristic as voxelization.ipynb).
    """
    bounds = points.max(axis=0) - points.min(axis=0)
    if voxel_size is None:
        # Use the *smallest* bounding extent to set voxel size,
        # capped between 1cm and 20cm.  For long narrow vine rows
        # using max(bounds) produces ~1m voxels which is too coarse.
        voxel_size = float(min(bounds) * 0.05)
        voxel_size = max(0.01, min(voxel_size, 0.20))

    min_bound = points.min(axis=0)
    indices = np.floor((points - min_bound) / voxel_size).astype(np.int64)
    unique_voxels = np.unique(indices, axis=0)
    n_voxels = len(unique_voxels)
    volume = n_voxels * (voxel_size ** 3)

    return {
        "vol_voxel": float(volume),
        "voxel_size": float(voxel_size),
        "n_voxels": int(n_voxels),
    }


# ---------------------------------------------------------------------------
# 3b. Slice + alpha-hull volume (from enhanced_volume_calculation.ipynb)
# ---------------------------------------------------------------------------

def _alpha_hull_area_2d(points_xy: np.ndarray, rmax: float) -> float:
    """
    Compute the area of a 2D alpha shape (concave hull) using Delaunay
    triangulation with a circumradius filter.
    """
    if len(points_xy) < 3:
        return 0.0

    try:
        tri = Delaunay(points_xy)
    except Exception:
        return 0.0

    simplices = tri.simplices
    A = points_xy[simplices[:, 0]]
    B = points_xy[simplices[:, 1]]
    C = points_xy[simplices[:, 2]]

    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(A - C, axis=1)
    c = np.linalg.norm(A - B, axis=1)

    s = 0.5 * (a + b + c)
    tri_area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0.0))
    R = (a * b * c) / np.maximum(4.0 * tri_area, 1e-12)

    keep = R <= rmax
    kept = simplices[keep]

    if len(kept) == 0:
        try:
            hull = ConvexHull(points_xy)
            return float(hull.volume)  # 2D: volume = area
        except Exception:
            return 0.0

    # Extract boundary edges
    edges = np.vstack([kept[:, [0, 1]], kept[:, [1, 2]], kept[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    _, inv, cnt = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    boundary_mask = cnt[inv] == 1
    boundary_edges = edges[boundary_mask]

    lines = [LineString([points_xy[i], points_xy[j]]) for i, j in boundary_edges]
    polys = list(polygonize(lines))

    if not polys:
        try:
            hull = ConvexHull(points_xy)
            return float(hull.volume)
        except Exception:
            return 0.0

    return float(sum(p.area for p in polys))


def compute_slice_volume(points: np.ndarray,
                         n_slices: int = 30,
                         rmax: float | None = None) -> dict:
    """
    Slice the point cloud along Z, compute alpha-hull area per slice,
    then integrate: V = sum(A_k * dz).

    If rmax is None, estimate from point spacing.
    """
    z = points[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    z_range = zmax - zmin

    if z_range < 0.01:
        return {"vol_slice": 0.0, "n_slices_used": 0}

    if rmax is None:
        # Estimate from mean nearest-neighbor distance in XY
        xy = points[:, :2]
        sample_n = min(len(xy), 5000)
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(xy), sample_n, replace=False)
        tree = cKDTree(xy)
        dists, _ = tree.query(xy[sample_idx], k=2)
        nn_dist = float(np.median(dists[:, 1]))
        rmax = max(nn_dist * 10, 0.5)

    edges = np.linspace(zmin, zmax, n_slices + 1)
    total_vol = 0.0
    slices_used = 0

    for k in range(n_slices):
        lo, hi = edges[k], edges[k + 1]
        if k < n_slices - 1:
            mask = (z >= lo) & (z < hi)
        else:
            mask = (z >= lo) & (z <= hi)

        s_pts = points[mask]
        if len(s_pts) < 10:
            continue

        dz = hi - lo
        area = _alpha_hull_area_2d(s_pts[:, :2], rmax)
        total_vol += area * dz
        slices_used += 1

    return {
        "vol_slice": float(total_vol),
        "n_slices_used": slices_used,
    }


# ---------------------------------------------------------------------------
# 4. NDVI statistics
# ---------------------------------------------------------------------------

def compute_ndvi_stats(las) -> dict:
    """
    Compute NDVI statistics from cluster LAS. Uses the pre-computed ndvi
    dimension if available, otherwise recomputes from red/nir.
    """
    dim_names = set(d.lower() for d in las.point_format.dimension_names)

    ndvi = None
    if "ndvi" in dim_names:
        ndvi = np.array(las.ndvi, dtype=np.float64)
    else:
        has_red = "red" in dim_names
        has_nir = "nir" in dim_names or "infrared" in dim_names
        if has_red and has_nir:
            red = np.array(las.red, dtype=np.float64)
            nir_field = "nir" if "nir" in dim_names else "infrared"
            nir = np.array(getattr(las, nir_field), dtype=np.float64)
            eps = 1e-6
            ndvi = (nir - red) / (nir + red + eps)

    if ndvi is None or len(ndvi) == 0:
        return {
            "ndvi_mean": None, "ndvi_std": None,
            "ndvi_p10": None, "ndvi_p90": None,
            "ndvi_low_frac": None, "ndvi_range": None,
        }

    p10 = float(np.percentile(ndvi, 10))
    p90 = float(np.percentile(ndvi, 90))
    return {
        "ndvi_mean": float(np.mean(ndvi)),
        "ndvi_std": float(np.std(ndvi)),
        "ndvi_p10": p10,
        "ndvi_p90": p90,
        "ndvi_low_frac": float(np.mean(ndvi < 0.2)),
        "ndvi_range": p90 - p10,
    }


# ---------------------------------------------------------------------------
# 5. CRS extraction
# ---------------------------------------------------------------------------

def extract_crs_wkt(las_path: Path) -> str | None:
    """
    Try to extract the CRS WKT string from LAS VLRs.
    Returns the raw WKT string or None.
    """
    try:
        with laspy.open(str(las_path)) as f:
            for vlr in f.header.vlrs:
                if hasattr(vlr, "string") and vlr.string:
                    return vlr.string.strip()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# 6. Parse cluster ID from filename
# ---------------------------------------------------------------------------

def _parse_cluster_id(filename: str) -> int | None:
    m = re.search(r"cluster_(\d+)", filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_cluster(las_path: Path, source_crs_wkt: str | None = None) -> dict:
    """Process a single cluster LAS file and return a feature dict."""
    las = laspy.read(str(las_path))
    points = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    n_points = len(points)

    if n_points < 10:
        print(f"  SKIP {las_path.name}: only {n_points} points")
        return {}

    cluster_id = _parse_cluster_id(las_path.stem)

    # Geometry
    geom = compute_row_geometry(points)

    # Height
    height = compute_slope_aware_height(points)

    # Volume (both methods)
    vol_vox = compute_voxel_volume(points)
    vol_slc = compute_slice_volume(points)

    # NDVI
    ndvi = compute_ndvi_stats(las)

    row_length = geom["row_length"]
    _safe = lambda num, denom: float(num / denom) if denom > 0 else float("nan")

    row = {
        "cluster_file": las_path.name,
        "row_id": cluster_id,
        "point_count": n_points,
        "points_per_m": _safe(n_points, row_length),
    }
    row.update(geom)
    row.update(height)
    row.update(vol_vox)
    row["vol_voxel_per_m"] = _safe(vol_vox["vol_voxel"], row_length)
    row.update(vol_slc)
    row.update(ndvi)

    if source_crs_wkt:
        row["crs_wkt"] = source_crs_wkt

    return row


def main():
    parser = argparse.ArgumentParser(description="Compute per-row features for vineyard clusters.")
    parser.add_argument("--in-dir", required=True, help="Directory with cluster *_ndvi.las files")
    parser.add_argument("--out", default=None, help="Output parquet path (default: <in-dir>/row_features.parquet)")
    parser.add_argument("--pattern", default="*_ndvi.las", help="Glob pattern for cluster files")
    parser.add_argument("--source-las", default=None, help="Source LAS to extract CRS from")
    parser.add_argument("--voxel-size", type=float, default=None, help="Voxel size in meters (default: auto)")
    parser.add_argument("--n-slices", type=int, default=30, help="Number of Z-slices for slice volume")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        print(f"ERROR: {in_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else in_dir / "row_features.parquet"

    cluster_files = sorted(in_dir.glob(args.pattern))
    if not cluster_files:
        print(f"No files matching '{args.pattern}' in {in_dir}", file=sys.stderr)
        sys.exit(1)

    # Extract CRS from source LAS if provided
    source_crs_wkt = None
    if args.source_las:
        source_crs_wkt = extract_crs_wkt(Path(args.source_las))
        if source_crs_wkt:
            print(f"CRS extracted from {args.source_las} ({len(source_crs_wkt)} chars)")
        else:
            print(f"WARNING: no CRS found in {args.source_las}")

    print(f"Processing {len(cluster_files)} cluster files from {in_dir}")
    rows = []
    for i, fp in enumerate(cluster_files):
        print(f"  [{i+1}/{len(cluster_files)}] {fp.name}")
        row = process_cluster(fp, source_crs_wkt)
        if row:
            rows.append(row)

    if not rows:
        print("No rows produced.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Sort by row_id if available
    if "row_id" in df.columns and df["row_id"].notna().all():
        df = df.sort_values("row_id").reset_index(drop=True)

    df.to_parquet(str(out_path), index=False)

    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nSummary:")
    summary_cols = [
        "row_id", "point_count", "points_per_m",
        "height_max", "height_mean", "height_std", "height_cv", "ground_slope_deg",
        "row_length", "row_width", "azimuth_deg",
        "vol_voxel", "vol_voxel_per_m", "vol_slice",
        "ndvi_mean", "ndvi_range",
    ]
    existing = [c for c in summary_cols if c in df.columns]
    print(df[existing].to_string(index=False))


if __name__ == "__main__":
    main()
