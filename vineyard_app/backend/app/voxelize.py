"""Server-side voxelisation of vineyard cluster point clouds.

Reuses the same occupancy rule as
`vegetation-pcd-analysis/scripts/compute_row_features.compute_voxel_volume`:

    idx = floor((p - p_min) / voxel_size)
    n_voxels = number of unique idx triplets
    volume = n_voxels * voxel_size**3

so the voxel count and volume reported here are byte-identical to the
production pipeline's `vol_voxel` column at the same `voxel_size`.

For each occupied voxel the function emits the **centre** of the voxel in
world coordinates, then re-centres the whole scene to the ground centroid
(same convention as `read_all_clusters` in data.py) so the front-end can
drop the cubes directly into the existing scene origin without re-mapping.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import laspy
import numpy as np


# Safety caps so a 0.005 m slider value can't accidentally ship 5 M cubes
# across the wire and crash the browser.
MIN_VOXEL_SIZE = 0.02   # metres — finer than this is shown as a clamped warning
MAX_VOXEL_SIZE = 1.00   # metres
MAX_VOXELS_PER_CLUSTER = 60_000   # per-cluster cap (random subsample if exceeded)
MAX_TOTAL_VOXELS = 300_000        # global cap across all clusters

_CLUSTER_ID_RE = re.compile(r"cluster_(\d+)")


def _cluster_id(name: str) -> Optional[int]:
    m = _CLUSTER_ID_RE.search(name)
    return int(m.group(1)) if m else None


def _voxelise(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, int]:
    """
    Returns (centres, n_voxels). `centres` is an (M, 3) float64 array of
    voxel centres in the same coordinate frame as the input points.
    """
    min_bound = points.min(axis=0)
    idx = np.floor((points - min_bound) / voxel_size).astype(np.int64)
    unique = np.unique(idx, axis=0)
    centres = (unique.astype(np.float64) + 0.5) * voxel_size + min_bound
    return centres, int(len(unique))


def voxelise_job(job_dir: Path, voxel_size: float, seed: int = 0) -> dict:
    """
    Build the voxel-grid payload for one finished job.

    Raises ValueError on out-of-range `voxel_size` so the API layer can
    convert it into a 422.
    """
    if not math.isfinite(voxel_size):
        raise ValueError(f"voxel_size must be finite (got {voxel_size})")
    if voxel_size < MIN_VOXEL_SIZE or voxel_size > MAX_VOXEL_SIZE:
        raise ValueError(
            f"voxel_size must be in [{MIN_VOXEL_SIZE}, {MAX_VOXEL_SIZE}] m "
            f"(got {voxel_size})"
        )

    clusters_dir = job_dir / "clusters"
    las_files = sorted(clusters_dir.glob("*_cluster_*_ndvi.las"))

    rng = np.random.default_rng(seed)
    per_cluster: list[dict] = []
    accumulated_world_centres: list[np.ndarray] = []
    total_n_voxels = 0
    total_volume = 0.0

    for f in las_files:
        las = laspy.read(str(f))
        n_points = int(len(las.x))
        if n_points == 0:
            continue

        pts = np.stack(
            [np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)],
            axis=-1,
        ).astype(np.float64)

        centres, n_v = _voxelise(pts, voxel_size)
        if n_v == 0:
            continue

        truncated = False
        if n_v > MAX_VOXELS_PER_CLUSTER:
            truncated = True
            keep = rng.choice(n_v, MAX_VOXELS_PER_CLUSTER, replace=False)
            centres_send = centres[keep]
        else:
            centres_send = centres

        vol = n_v * (voxel_size ** 3)
        total_n_voxels += n_v
        total_volume += vol

        per_cluster.append({
            "row_id": _cluster_id(f.name),
            "file": f.name,
            "n_voxels": n_v,
            "n_voxels_sent": int(len(centres_send)),
            "vol_voxel": float(vol),
            "n_points": n_points,
            "truncated": truncated,
            "_world_centres": centres_send,  # internal, removed before return
        })
        accumulated_world_centres.append(centres_send)

    if not per_cluster:
        return {
            "voxel_size": float(voxel_size),
            "center": [0.0, 0.0, 0.0],
            "clusters": [],
            "total_voxels": 0,
            "total_volume": 0.0,
            "min_voxel_size": MIN_VOXEL_SIZE,
            "max_voxel_size": MAX_VOXEL_SIZE,
            "exceeded_total_cap": False,
        }

    # Global cap: if the union of all per-cluster surviving centres still
    # exceeds MAX_TOTAL_VOXELS, drop down proportionally.
    grand_total_sent = sum(c["n_voxels_sent"] for c in per_cluster)
    exceeded = grand_total_sent > MAX_TOTAL_VOXELS
    if exceeded:
        factor = MAX_TOTAL_VOXELS / grand_total_sent
        for c in per_cluster:
            world = c.pop("_world_centres")
            keep_n = max(1, int(round(c["n_voxels_sent"] * factor)))
            if keep_n < len(world):
                idx = rng.choice(len(world), keep_n, replace=False)
                world = world[idx]
            c["_world_centres"] = world
            c["n_voxels_sent"] = int(len(world))
            c["truncated"] = True

    combined = np.concatenate(
        [c["_world_centres"] for c in per_cluster], axis=0
    )
    cx = float(combined[:, 0].mean())
    cy = float(combined[:, 1].mean())
    cz = float(combined[:, 2].min())  # match data.read_all_clusters: ground anchor

    offset = np.array([cx, cy, cz], dtype=np.float64)

    out_clusters: list[dict] = []
    for c in per_cluster:
        world = c.pop("_world_centres")
        recentred = (world - offset).astype(np.float32)
        c["xyz"] = recentred.reshape(-1).tolist()
        out_clusters.append(c)

    return {
        "voxel_size": float(voxel_size),
        "center": [cx, cy, cz],
        "clusters": out_clusters,
        "total_voxels": int(total_n_voxels),
        "total_volume": float(total_volume),
        "min_voxel_size": MIN_VOXEL_SIZE,
        "max_voxel_size": MAX_VOXEL_SIZE,
        "exceeded_total_cap": bool(exceeded),
    }
