import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

try:
    import pdal  # type: ignore
    _HAS_PDAL_BINDINGS = True
except Exception:  # pragma: no cover - optional dependency
    pdal = None
    _HAS_PDAL_BINDINGS = False

try:
    import laspy  # type: ignore
except Exception as exc:
    raise ImportError("laspy is required for this script. Please install it with `pip install laspy`.") from exc

try:
    import open3d as o3d  # type: ignore
except Exception as exc:
    raise ImportError("open3d is required for visualization. Please install it with `pip install open3d`.") from exc


DEFAULT_SMRF_PARAMS: Dict[str, float] = {
    "slope": 0.15,
    "window": 16.0,
    "threshold": 0.5,
    "scalar": 1.25,
}


def _build_pdal_pipeline_dict(input_path: str, output_path: str, smrf_params: Dict[str, float]) -> Dict:
    """Build an in-memory PDAL pipeline dict for SMRF ground classification."""
    params = {**DEFAULT_SMRF_PARAMS, **(smrf_params or {})}

    return {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_path,
                # Ignore any SRS/WKT in the input header to avoid GDAL WKT parsing issues
                "nosrs": True,
            },
            {
                "type": "filters.smrf",
                "slope": float(params["slope"]),
                "window": float(params["window"]),
                "threshold": float(params["threshold"]),
                "scalar": float(params["scalar"]),
            },
            {
                "type": "writers.las",
                "filename": output_path,
                "minor_version": 4,
                "forward": "all",
                "extra_dims": "all"
            },
        ]
    }


def _run_pdal_pipeline_bindings(pipeline_dict: Dict) -> None:
    """Run a PDAL pipeline using Python bindings."""
    if not _HAS_PDAL_BINDINGS:
        raise RuntimeError("PDAL Python bindings are not available.")

    pipeline_json = json.dumps(pipeline_dict)
    pl = pdal.Pipeline(pipeline_json)
    _ = pl.execute()


def _run_pdal_pipeline_cli(pipeline_dict: Dict) -> None:
    """Run a PDAL pipeline using the `pdal pipeline` CLI as a fallback."""
    if shutil.which("pdal") is None:
        raise RuntimeError("Neither PDAL Python bindings nor `pdal` CLI are available.")

    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(pipeline_dict, tmp)
        tmp_path = tmp.name

    try:
        subprocess.run(["pdal", "pipeline", tmp_path], check=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def run_smrf_classification(
    input_las: str,
    out_dir: str = "./scripts/out_ground",
    smrf_params: Optional[Dict[str, float]] = None,
) -> str:
    """Run PDAL SMRF ground classification on a LAS/LAZ file.

    Parameters
    ----------
    input_las : str
        Path to input LAS/LAZ file.
    out_dir : str, optional
        Directory to store the classified output, by default "./scripts/out_ground".
    smrf_params : dict, optional
        Dictionary with SMRF parameters (slope, window, threshold, scalar).

    Returns
    -------
    str
        Path to the classified LAS/LAZ file.
    """
    input_path = Path(input_las)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    classified_path = out_dir_path / f"{input_path.stem}_classified_smrf{input_path.suffix}"

    pipeline_dict = _build_pdal_pipeline_dict(str(input_path), str(classified_path), smrf_params or {})

    if _HAS_PDAL_BINDINGS:
        _run_pdal_pipeline_bindings(pipeline_dict)
    else:
        _run_pdal_pipeline_cli(pipeline_dict)

    if not classified_path.is_file():
        raise RuntimeError(f"PDAL did not produce the expected output file: {classified_path}")

    return str(classified_path)


def split_ground_non_ground(
    classified_las: str,
    out_dir: str,
) -> Tuple[str, str]:
    """Split a classified LAS/LAZ file into ground (Classification==2) and
    non-ground (Classification!=2) point clouds and write them to disk.

    Parameters
    ----------
    classified_las : str
        Path to SMRF-classified LAS/LAZ file.
    out_dir : str
        Directory where ground and non-ground files will be written.

    Returns
    -------
    (str, str)
        Paths to the ground and non-ground LAS files respectively.
    """
    in_path = Path(classified_las)
    if not in_path.is_file():
        raise FileNotFoundError(f"Classified file does not exist: {in_path}")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    with laspy.open(in_path) as src:
        hdr = src.header
        points = src.read()

    total_points = len(points.x)

    if not hasattr(points, "classification"):
        print("Warning: input LAS does not contain 'Classification' data; treating all points as non-ground.")
        classification = np.zeros(total_points, dtype=np.uint8)
    else:
        classification = np.array(points.classification)
    ground_mask = classification == 2
    non_ground_mask = ~ground_mask

    ground_count = int(ground_mask.sum())
    non_ground_count = int(non_ground_mask.sum())

    removed_count = ground_count  # interpreting removal as removing ground points
    if total_points > 0:
        ground_pct = 100.0 * ground_count / total_points
        non_ground_pct = 100.0 * non_ground_count / total_points
        removed_pct = 100.0 * removed_count / total_points
    else:
        ground_pct = non_ground_pct = removed_pct = 0.0

    print("=== SMRF Ground Classification Summary ===")
    print(f"Total points (original): {total_points}")
    print(f"Ground points (Classification==2): {ground_count} ({ground_pct:.2f}%)")
    print(f"Non-ground points (Classification!=2): {non_ground_count} ({non_ground_pct:.2f}%)")
    print(f"'Removed' (ground) points: {removed_count} ({removed_pct:.2f}%)")

    # Prepare output file paths
    ground_out = out_dir_path / f"{in_path.stem}_ground{in_path.suffix}"
    non_ground_out = out_dir_path / f"{in_path.stem}_non_ground{in_path.suffix}"

    # Write ground points
    ground_data = laspy.LasData(hdr)
    ground_data.points = points.points[ground_mask]
    ground_data.write(ground_out)

    # Write non-ground points
    non_ground_data = laspy.LasData(hdr)
    non_ground_data.points = points.points[non_ground_mask]
    non_ground_data.write(non_ground_out)

    return str(ground_out), str(non_ground_out)


def _las_to_o3d_point_cloud(las_path: str, color: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """Load LAS/LAZ into an Open3D point cloud with a single RGB color."""
    with laspy.open(las_path) as src:
        pts = src.read()

    xyz = np.vstack((pts.x, pts.y, pts.z)).T.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    colors = np.tile(np.array(color, dtype=np.float64), (xyz.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_before_after(
    original_las: str,
    non_ground_las: str,
    ground_las: Optional[str] = None,
) -> None:
    """Visualize before/after SMRF classification with Open3D.

    - Before: original point cloud
    - After: non-ground only
    - Optional comparison: ground vs non-ground side by side (translated)
    """
    if not Path(original_las).is_file():
        raise FileNotFoundError(f"Original file does not exist: {original_las}")
    if not Path(non_ground_las).is_file():
        raise FileNotFoundError(f"Non-ground file does not exist: {non_ground_las}")

    print("Loading point clouds for visualization...")

    # Colors: original=light gray, non-ground=blue, ground=brown
    pcd_original = _las_to_o3d_point_cloud(original_las, (0.8, 0.8, 0.8))
    pcd_non_ground = _las_to_o3d_point_cloud(non_ground_las, (0.2, 0.4, 0.9))

    print("Showing overlay: original (gray) and non-ground (blue)...")
    o3d.visualization.draw_geometries([
        pcd_original,
        pcd_non_ground,
    ])

    if ground_las is not None and Path(ground_las).is_file():
        pcd_ground = _las_to_o3d_point_cloud(ground_las, (0.6, 0.3, 0.1))

        bbox = pcd_non_ground.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()[0]
        translate_vec = np.array([extent * 2.0, 0.0, 0.0])

        pcd_non_ground_shifted = pcd_non_ground.translate(translate_vec)

        print("Showing side-by-side: ground (brown) vs non-ground (blue, shifted)...")
        o3d.visualization.draw_geometries([
            pcd_ground,
            pcd_non_ground_shifted,
        ])


if __name__ == "__main__":
    example_input = "../datasource/2025-07-15-MS_Vinograd_1.las"
    output_directory = "./out_ground"

    classified = run_smrf_classification(
        example_input,
        out_dir=output_directory,
        smrf_params={
            "slope": 0.15,
            "window": 16.0,
            "threshold": 0.5,
            "scalar": 1.25,
        },
    )

    ground_file, non_ground_file = split_ground_non_ground(classified, output_directory)

    visualize_before_after(example_input, non_ground_file, ground_file)

