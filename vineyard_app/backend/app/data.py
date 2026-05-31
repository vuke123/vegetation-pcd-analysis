"""Read parquet metrics and cluster LAS files into API-friendly payloads."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional

import laspy
import numpy as np
import pandas as pd

from .config import MAX_POINTS_PER_CLUSTER

_CLUSTER_ID_RE = re.compile(r"cluster_(\d+)")


def _cluster_id_from_name(name: str) -> Optional[int]:
    m = _CLUSTER_ID_RE.search(name)
    return int(m.group(1)) if m else None


def _clean(value):
    """Convert numpy/NaN values into JSON-safe primitives."""
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bytes,)):
        return value.decode("utf-8", errors="replace")
    return value


def read_metrics(job_dir: Path) -> list[dict]:
    parquet = job_dir / "clusters" / "row_features.parquet"
    if not parquet.exists():
        return []
    df = pd.read_parquet(parquet)
    if "crs_wkt" in df.columns:
        df = df.drop(columns=["crs_wkt"])  # huge string, not needed in UI
    records: list[dict] = []
    for raw in df.to_dict(orient="records"):
        records.append({k: _clean(v) for k, v in raw.items()})
    return records


def _downsample(n: int, target: int) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    stride = int(math.ceil(n / target))
    return np.arange(0, n, stride)


def read_cluster_points(las_path: Path, max_points: int) -> dict:
    """Load a single cluster LAS and return downsampled xyz + optional NDVI."""
    las = laspy.read(str(las_path))
    n = len(las.x)
    if n == 0:
        return {"count": 0, "xyz": [], "ndvi": []}

    idx = _downsample(n, max_points)
    x = np.asarray(las.x)[idx].astype(np.float32)
    y = np.asarray(las.y)[idx].astype(np.float32)
    z = np.asarray(las.z)[idx].astype(np.float32)

    ndvi = None
    for candidate in ("ndvi", "NDVI", "Ndvi"):
        if candidate in las.point_format.dimension_names:
            ndvi = np.asarray(las[candidate])[idx].astype(np.float32)
            break

    xyz = np.stack([x, y, z], axis=1).reshape(-1).tolist()
    return {
        "count": int(len(idx)),
        "original_count": int(n),
        "xyz": xyz,
        "ndvi": ndvi.tolist() if ndvi is not None else None,
    }


def read_all_clusters(job_dir: Path, max_points: int = MAX_POINTS_PER_CLUSTER) -> dict:
    clusters_dir = job_dir / "clusters"
    las_files = sorted(clusters_dir.glob("*_cluster_*_ndvi.las"))

    clusters = []
    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    for f in las_files:
        row_id = _cluster_id_from_name(f.name)
        payload = read_cluster_points(f, max_points)
        if payload["count"] == 0:
            continue
        xyz = payload["xyz"]
        all_x.extend(xyz[0::3])
        all_y.extend(xyz[1::3])
        all_z.extend(xyz[2::3])
        clusters.append({
            "row_id": row_id,
            "file": f.name,
            "count": payload["count"],
            "original_count": payload["original_count"],
            "xyz": xyz,
            "ndvi": payload["ndvi"],
        })

    if not clusters:
        return {"center": [0, 0, 0], "clusters": []}

    cx = float(np.mean(all_x))
    cy = float(np.mean(all_y))
    cz = float(np.min(all_z))  # anchor to ground level, more natural framing

    # Re-center cluster points client-side — do it here to keep frontend simple.
    for c in clusters:
        xyz = c["xyz"]
        for i in range(0, len(xyz), 3):
            xyz[i] -= cx
            xyz[i + 1] -= cy
            xyz[i + 2] -= cz

    return {"center": [cx, cy, cz], "clusters": clusters}
