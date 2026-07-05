import numpy as np
import pandas as pd
import laspy
from pathlib import Path

# ----------------------------
# Sampling helpers (uniform)
# ----------------------------

def sample_points_in_sphere(n: int, radius: float, center: tuple[float, float, float], seed: int) -> np.ndarray:
    """Uniformly sample points inside a solid sphere."""
    rng = np.random.default_rng(seed)

    # Random direction (uniform on the sphere): normal distribution + normalization
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    # Uniform over volume: r = R * U^(1/3)
    r = radius * np.cbrt(rng.random(n))
    pts = v * r[:, None] + np.array(center)[None, :]
    return pts

def sample_points_in_cylinder(n: int, radius: float, height: float, center: tuple[float, float, float],
                              axis: str = "z", seed: int = 0) -> np.ndarray:
    """
    Uniformly sample points inside a solid cylinder.
    axis: "x", "y" or "z" (cylinder axis)
    height: total height (from -h/2 to +h/2)
    """
    rng = np.random.default_rng(seed)

    # Uniform in the disk: r = R * sqrt(U), theta = 2piV
    u = rng.random(n)
    v = rng.random(n)
    rr = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v

    a = rr * np.cos(theta)
    b = rr * np.sin(theta)
    t = (rng.random(n) - 0.5) * height

    pts = np.zeros((n, 3), dtype=np.float64)

    if axis == "z":
        pts[:, 0] = a
        pts[:, 1] = b
        pts[:, 2] = t
    elif axis == "y":
        pts[:, 0] = a
        pts[:, 2] = b
        pts[:, 1] = t
    elif axis == "x":
        pts[:, 1] = a
        pts[:, 2] = b
        pts[:, 0] = t
    else:
        raise ValueError("axis must be one of: 'x','y','z'")

    pts += np.array(center)[None, :]
    return pts

# ----------------------------
# Volumes (ground truth)
# ----------------------------

def volume_sphere(r: float) -> float:
    return (4.0 / 3.0) * np.pi * (r ** 3)

def volume_cylinder(r: float, h: float) -> float:
    return np.pi * (r ** 2) * h

def y_extent_radius(obj: dict) -> float:
    """
    How far the object extends in the +/-Y direction around its center (bounding extent in Y).
    - sphere: r
    - cylinder (axis z or x): r
    - cylinder (axis y): height/2
    """
    if obj["type"] == "sphere":
        return float(obj["radius"])
    if obj["type"] == "cylinder":
        axis = obj.get("axis", "z")
        if axis == "y":
            return float(obj["height"]) / 2.0
        return float(obj["radius"])
    raise ValueError("Unknown object type")

# ----------------------------
# LAS writer
# ----------------------------

def write_las(points_xyz: np.ndarray, classification: np.ndarray, intensity: np.ndarray,
              out_path: str, scale: float = 0.001) -> str:
    """
    Save XYZ + classification + intensity to LAS.
    scale=0.001 => mm resolution.
    """
    out_path = str(Path(out_path))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([scale, scale, scale], dtype=np.float64)

    mins = points_xyz.min(axis=0)
    header.offsets = mins.astype(np.float64)

    las = laspy.LasData(header)
    las.x = points_xyz[:, 0]
    las.y = points_xyz[:, 1]
    las.z = points_xyz[:, 2]
    las.classification = classification.astype(np.uint8)  # 1..4 (object ID)
    las.intensity = intensity.astype(np.uint16)           # 1=cylinder, 2=sphere
    las.write(out_path)
    return out_path

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    # Output
    OUT_LAS = "synthetic_row_overlap.las"
    OUT_CSV = "synthetic_row_overlap_ground_truth.csv"

    # LAS quantization
    SCALE = 0.001  # 1 mm

    # Point density (approx) - points per cubic meter
    POINTS_PER_M3 = 80_000

    # Overlap control:
    # 0.15 => neighbours overlap by ~15% of the sum of their Y-extents
    OVERLAP_RATIO = 0.20  # try 0.10–0.30

    # Slight jitter along X (so they are not perfectly on the same axis) - optional
    # Set to 0.0 to keep them exactly on the same X axis
    JITTER_X_METERS = 0.05  # +/- 5 cm
    JITTER_Z_METERS = 0.00  # e.g. 0.03 for +/- 3 cm

    SEED_LAYOUT = 123

    # Object row definition: cylinder – sphere – sphere (different size) – cylinder
    # All are "solid" (points inside the volume).
    objects = [
        {"name": "cyl_1", "type": "cylinder", "radius": 0.35, "height": 1.80, "axis": "z"},
        {"name": "sph_1", "type": "sphere",   "radius": 0.55},
        {"name": "sph_2", "type": "sphere",   "radius": 0.35},
        {"name": "cyl_2", "type": "cylinder", "radius": 0.25, "height": 2.20, "axis": "z"},
    ]

    rng_layout = np.random.default_rng(SEED_LAYOUT)

    all_pts = []
    all_cls = []
    all_int = []
    gt_rows = []

    base_x, base_y, base_z = 0.0, 0.0, 0.0
    prev_center_y = None
    prev_yext = None

    for obj_id, obj in enumerate(objects, start=1):
        curr_yext = y_extent_radius(obj)

        # centers along Y with controlled overlap of the bounding extent
        if prev_center_y is None:
            cy = base_y
        else:
            d = (prev_yext + curr_yext) * (1.0 - OVERLAP_RATIO)
            cy = prev_center_y + d

        # small jitter along X/Z (optional)
        cx = base_x + (rng_layout.random() - 0.5) * 2.0 * JITTER_X_METERS
        cz = base_z + (rng_layout.random() - 0.5) * 2.0 * JITTER_Z_METERS

        center = (cx, cy, cz)

        # volume + number of points proportional to volume
        if obj["type"] == "sphere":
            r = float(obj["radius"])
            vol = volume_sphere(r)
            n = max(5_000, int(vol * POINTS_PER_M3))
            pts = sample_points_in_sphere(n=n, radius=r, center=center, seed=1000 + obj_id)
            intensity_val = 2  # sphere
        elif obj["type"] == "cylinder":
            r = float(obj["radius"])
            h = float(obj["height"])
            axis = obj.get("axis", "z")
            vol = volume_cylinder(r, h)
            n = max(5_000, int(vol * POINTS_PER_M3))
            pts = sample_points_in_cylinder(n=n, radius=r, height=h, center=center, axis=axis, seed=2000 + obj_id)
            intensity_val = 1  # cylinder
        else:
            raise ValueError(f"Unknown object type: {obj['type']}")

        cls = np.full((pts.shape[0],), obj_id, dtype=np.uint8)
        inten = np.full((pts.shape[0],), intensity_val, dtype=np.uint16)

        all_pts.append(pts)
        all_cls.append(cls)
        all_int.append(inten)

        gt_rows.append({
            "object_id": obj_id,
            "name": obj["name"],
            "type": obj["type"],
            "center_x": center[0],
            "center_y": center[1],
            "center_z": center[2],
            "radius_m": obj["radius"],
            "height_m": obj.get("height", np.nan),
            "axis": obj.get("axis", ""),
            "points_generated": int(pts.shape[0]),
            "ground_truth_volume_m3": float(vol),
        })

        prev_center_y = cy
        prev_yext = curr_yext

    points_xyz = np.vstack(all_pts)
    classification = np.concatenate(all_cls)
    intensity = np.concatenate(all_int)

    # Save LAS
    las_path = write_las(points_xyz, classification, intensity, OUT_LAS, scale=SCALE)

    # Save GT CSV (with TOTAL row)
    gt_df = pd.DataFrame(gt_rows)
    total_vol = float(gt_df["ground_truth_volume_m3"].sum())

    gt_df.loc[len(gt_df)] = {
        "object_id": 0,
        "name": "TOTAL",
        "type": "all",
        "center_x": np.nan, "center_y": np.nan, "center_z": np.nan,
        "radius_m": np.nan, "height_m": np.nan, "axis": "",
        "points_generated": int(points_xyz.shape[0]),
        "ground_truth_volume_m3": total_vol,
    }

    gt_df.to_csv(OUT_CSV, index=False)

    print("Saved LAS:", las_path)
    print("Saved GT :", OUT_CSV)
    print("Total points:", points_xyz.shape[0])
    print("Total ground-truth volume (m^3):", total_vol)
    print("Overlap ratio:", OVERLAP_RATIO)
    print("Jitter X/Z (m):", JITTER_X_METERS, "/", JITTER_Z_METERS)
