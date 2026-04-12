"""
LAS / LAZ I/O utilities.

Reads point clouds with laspy into numpy arrays, handling missing dimensions
gracefully. Designed to work with multispectral drone LAS files from the
existing vineyard pipeline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import laspy
import numpy as np

logger = logging.getLogger(__name__)

# Dimensions that may exist in multispectral vineyard LAS files.
KNOWN_DIMS = {
    "x", "y", "z",
    "red", "green", "blue",
    "nir", "infrared", "near_infrared",
    "ndvi",
    "intensity",
    "return_number", "number_of_returns",
    "classification", "label",
    "normal_x", "normal_y", "normal_z",
}


def read_las(path: str | Path) -> dict:
    """
    Read a LAS/LAZ file and return a dict of numpy arrays.

    Returns
    -------
    dict with keys:
        'xyz'    : (N, 3) float64
        'dims'   : dict of dim_name -> (N,) array for all available dimensions
        'n_points': int
        'path'   : str
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LAS file not found: {path}")

    las = laspy.read(str(path))
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)

    dims = {}
    for name in las.point_format.dimension_names:
        try:
            arr = np.asarray(las[name])
            dims[name.lower()] = arr
        except Exception:
            pass

    logger.debug("Read %s: %d points, dims=%s", path.name, len(xyz), list(dims.keys()))

    return {
        "xyz": xyz,
        "dims": dims,
        "n_points": len(xyz),
        "path": str(path),
    }


def list_available_features(path: str | Path) -> list[str]:
    """Return the list of dimension names available in a LAS file."""
    path = Path(path)
    with laspy.open(str(path)) as f:
        return [d.lower() for d in f.header.point_format.dimension_names]


def extract_labels(dims: dict) -> np.ndarray | None:
    """
    Try to extract semantic labels from LAS dimensions.

    Looks for 'label', then 'classification'. Returns None if neither exists.
    """
    for key in ("label", "classification"):
        if key in dims:
            return dims[key].astype(np.int32)
    return None


def write_las_with_labels(xyz: np.ndarray,
                          labels: np.ndarray,
                          out_path: str | Path,
                          extra_dims: dict | None = None) -> None:
    """
    Write a LAS file with XYZ coordinates and a 'label' extra dimension.

    Useful for saving inference predictions as point clouds that can be
    opened in CloudCompare.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = laspy.LasHeader(point_format=0, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="label", type=np.uint8))

    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las["label"] = labels.astype(np.uint8)

    if extra_dims:
        for name, arr in extra_dims.items():
            if name not in ("x", "y", "z", "label"):
                try:
                    header.add_extra_dim(
                        laspy.ExtraBytesParams(name=name, type=arr.dtype)
                    )
                    las[name] = arr
                except Exception:
                    logger.warning("Could not add extra dim %r", name)

    las.write(str(out_path))
    logger.info("Wrote %d points to %s", len(xyz), out_path)
