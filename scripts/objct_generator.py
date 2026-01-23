import numpy as np
import pandas as pd
import laspy
from pathlib import Path

# ----------------------------
# Sampling helpers (uniform)
# ----------------------------

def sample_points_in_sphere(n: int, radius: float, center: tuple[float, float, float], seed: int) -> np.ndarray:
    """Uniformno uzorkuje točke unutar kugle (solid)."""
    rng = np.random.default_rng(seed)

    # Random smjer (uniformno po sferi): normal distribucija + normalizacija
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    # Uniformno po volumenu: r = R * U^(1/3)
    r = radius * np.cbrt(rng.random(n))
    pts = v * r[:, None] + np.array(center)[None, :]
    return pts

def sample_points_in_cylinder(n: int, radius: float, height: float, center: tuple[float, float, float],
                              axis: str = "z", seed: int = 0) -> np.ndarray:
    """
    Uniformno uzorkuje točke unutar punog valjka.
    axis: "x", "y" ili "z" (os valjka)
    height: ukupna visina (od -h/2 do +h/2)
    """
    rng = np.random.default_rng(seed)

    # Uniformno u disku: r = R * sqrt(U), theta = 2piV
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
    Koliko objekt "strši" u +/-Y smjeru oko centra (bounding extent u Y).
    - sphere: r
    - cylinder (axis z ili x): r
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
    Spremi XYZ + classification + intensity u LAS.
    scale=0.001 => mm rezolucija.
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
    las.classification = classification.astype(np.uint8)  # 1..4 (ID objekta)
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
    # 0.15 => susjedi se preklapaju oko 15% njihovih "Y-extent" zbroja
    OVERLAP_RATIO = 0.20  # probaj 0.10–0.30

    # Lagani jitter po X (da ne budu savršeno u istoj osi) - opcionalno
    # Stavi 0.0 ako želiš baš na istoj X osi
    JITTER_X_METERS = 0.05  # +/- 5 cm
    JITTER_Z_METERS = 0.00  # npr. 0.03 za +/- 3 cm

    SEED_LAYOUT = 123

    # Definicija objekata u nizu: cylinder – sphere – sphere (druga veličina) – cylinder
    # Svi su "solid" (točke unutar volumena).
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

        # centri po Y s kontroliranim preklapanjem bounding extenta
        if prev_center_y is None:
            cy = base_y
        else:
            d = (prev_yext + curr_yext) * (1.0 - OVERLAP_RATIO)
            cy = prev_center_y + d

        # mali jitter po X/Z (opcionalno)
        cx = base_x + (rng_layout.random() - 0.5) * 2.0 * JITTER_X_METERS
        cz = base_z + (rng_layout.random() - 0.5) * 2.0 * JITTER_Z_METERS

        center = (cx, cy, cz)

        # volumen + broj točaka proporcionalno volumenu
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
