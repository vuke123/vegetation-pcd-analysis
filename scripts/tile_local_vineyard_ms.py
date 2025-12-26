"""XY-tiling visualization for LOCAL_VINEYARD_MS.las.

Loads the LAS file using the existing point_cloud_to_dataframe logic,
partitions the XY plane into 10 tiles, visualizes each tile separately,
then shows the full point cloud colored by tile id.
"""

import os
import subprocess
import glob

import laspy
import numpy as np
import open3d as o3d
import pandas as pd


# Path to the LAS file (same relative path as used in rasterizing.ipynb)
LAS_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../datasource/LOCAL_VINEYARD_MS.las")
)
OUTPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "out_tiles")
)
TILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "tiles")
)


def point_cloud_to_dataframe(file_path: str, apply_scale: bool = True) -> pd.DataFrame:
    """Dispatcher that calls the LAS loader implementation.

    This mirrors the logic from scripts/read_plot_voxelization.ipynb.
    """
    file_ext = file_path.lower().split(".")[-1]

    if file_ext in ["las", "laz"]:
        return _las_to_dataframe(file_path, apply_scale=apply_scale)
    else:
        raise ValueError(
            f"Unsupported file format: .{file_ext}. Supported: .las, .laz"
        )


def _las_to_dataframe(las_path: str, apply_scale: bool = True) -> pd.DataFrame:
    """Loads .las/.laz file and converts it to a pandas DataFrame.

    Implementation copied from scripts/read_plot_voxelization.ipynb so that
    this script can reuse the same loader behavior.
    """
    print(f"📂 Loading LAS/LAZ file: {las_path}")
    las = laspy.read(las_path)

    print(f"Number of points: {las.header.point_count:,}")

    dims = [dim.name for dim in las.point_format.dimensions]
    print(f"Detected dimensions: {dims}")

    data = {}
    for dim in dims:
        try:
            arr = getattr(las, dim)
            data[dim] = np.array(arr)
        except AttributeError:
            print(f"Field '{dim}' is not found.")
            continue

    df = pd.DataFrame(data)
    if apply_scale:
        sx, sy, sz = las.header.scales
        ox, oy, oz = las.header.offsets

        if "X" in df.columns:
            df["X"] = df["X"].astype(np.float64) * sx + ox
        if "Y" in df.columns:
            df["Y"] = df["Y"].astype(np.float64) * sy + oy
        if "Z" in df.columns:
            df["Z"] = df["Z"].astype(np.float64) * sz + oz

        print("Applied LAS header scale+offset to X,Y,Z.")
    else:
        print("Kept raw LAS X,Y,Z (no header scale+offset applied).")

    print(f"DataFrame created. Shape: {df.shape}")
    return df


def _choose_grid_for_tiles(n_tiles: int, width_x: float, width_y: float) -> tuple[int, int]:
    """Pick (nx, ny) with nx * ny == n_tiles, roughly matching XY aspect ratio.

    This keeps tiles more square when possible.
    """
    if n_tiles <= 0:
        raise ValueError("n_tiles must be positive")

    # Fallback aspect ratio if width_y is zero
    if width_y <= 0:
        target_ratio = 1.0
    else:
        target_ratio = width_x / max(width_y, 1e-9)

    best_nx, best_ny = 1, n_tiles
    best_diff = None

    for nx in range(1, n_tiles + 1):
        if n_tiles % nx != 0:
            continue
        ny = n_tiles // nx
        grid_ratio = nx / max(ny, 1e-9)
        diff = abs(grid_ratio - target_ratio)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_nx, best_ny = nx, ny

    print(f"Using grid {best_nx} x {best_ny} (nx x ny) for {n_tiles} tiles.")
    return best_nx, best_ny


def assign_xy_tiles(df: pd.DataFrame, n_tiles: int = 10) -> tuple[pd.DataFrame, int, int]:
    """Assign each point to one of n_tiles based on its (X, Y) position.

    The XY bounding box is split into a regular grid (nx * ny = n_tiles).
    All points in the same XY cell share the same tile id, regardless of Z.
    """
    if "X" not in df.columns or "Y" not in df.columns:
        raise KeyError("DataFrame must contain 'X' and 'Y' columns for tiling.")

    x = df["X"].to_numpy(dtype=np.float64)
    y = df["Y"].to_numpy(dtype=np.float64)

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    width_x = xmax - xmin
    width_y = ymax - ymin

    nx, ny = _choose_grid_for_tiles(n_tiles, width_x, width_y)

    # Avoid zero-width divisions
    dx = width_x / nx if width_x > 0 else 1.0
    dy = width_y / ny if width_y > 0 else 1.0

    # Normalize to [0, nx) and [0, ny), then floor
    ix = np.floor((x - xmin) / dx).astype(int)
    iy = np.floor((y - ymin) / dy).astype(int)

    # Handle potential numerical issues at the max boundary
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)

    tile_id = iy * nx + ix

    df_tiled = df.copy()
    df_tiled["tile_id"] = tile_id

    print(
        f"Assigned tiles: nx={nx}, ny={ny}, min_id={tile_id.min()}, max_id={tile_id.max()}"
    )
    return df_tiled, nx, ny


def dataframe_to_o3d_point_cloud(df: pd.DataFrame, color: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    """Convert a DataFrame with X, Y, Z (and optionally RGB) to an Open3D PointCloud.

    If `color` is provided, it must be a length-3 array with values in [0, 1] and
    is broadcast to all points. Otherwise, RGB columns are used if available.
    """
    required_cols = {"X", "Y", "Z"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise KeyError(f"DataFrame is missing required columns: {missing}")

    pts = df[["X", "Y", "Z"]].to_numpy(dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if color is not None:
        colors = np.tile(np.asarray(color, dtype=np.float64), (pts.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif {"red", "green", "blue"}.issubset(df.columns):
        rgb = df[["red", "green", "blue"]].to_numpy(dtype=np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def _extract_rgb_u8(df: pd.DataFrame) -> np.ndarray | None:
    """Return RGB values as uint8-like array (N,3) or None if RGB is missing.

    If the source is 16-bit (0..65535), values are approximately shifted to 8-bit
    by dividing by 256, mimicking the C++ behavior.
    """
    if not {"red", "green", "blue"}.issubset(df.columns):
        return None

    rgb = df[["red", "green", "blue"]].to_numpy(dtype=np.float32)
    if rgb.size == 0:
        return rgb

    max_val = float(np.max(rgb))
    if max_val > 255.0:
        rgb = rgb / 256.0

    rgb = np.clip(rgb, 0.0, 255.0)
    return rgb


def _rgb_to_hsv_np(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized RGB->HSV conversion.

    Input rgb_u8 is (N,3) in [0,255]. Output H in [0,360), S,V in [0,1].
    """
    if rgb_u8.size == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    rgb = rgb_u8.astype(np.float32) / 255.0
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    cmax = np.max(rgb, axis=1)
    cmin = np.min(rgb, axis=1)
    delta = cmax - cmin

    v = cmax
    s = np.zeros_like(v)
    nonzero_v = cmax > 1e-6
    s[nonzero_v] = delta[nonzero_v] / cmax[nonzero_v]

    h = np.zeros_like(v)
    nonzero_delta = delta > 1e-6

    mask_r = nonzero_delta & (cmax == r)
    mask_g = nonzero_delta & (cmax == g)
    mask_b = nonzero_delta & (cmax == b)

    h[mask_r] = 60.0 * np.mod(((g[mask_r] - b[mask_r]) / delta[mask_r]), 6.0)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0)

    h[h < 0.0] += 360.0
    return h, s, v


def _ground_like_color_mask(df: pd.DataFrame) -> np.ndarray:
    """Approximate C++ isGroundLikeColor for all rows, returning a boolean mask."""
    rgb_u8 = _extract_rgb_u8(df)
    if rgb_u8 is None or rgb_u8.size == 0:
        return np.zeros(len(df), dtype=bool)

    h, s, v = _rgb_to_hsv_np(rgb_u8)

    veg_green = (h >= 70.0) & (h <= 170.0) & (s > 0.20) & (v > 0.20)

    is_ground = np.zeros_like(v, dtype=bool)
    is_ground |= v < 0.16
    is_ground |= (s < 0.14) & (v > 0.18) & (v < 0.85)
    is_ground |= (h >= 8.0) & (h <= 70.0) & (s >= 0.12) & (v >= 0.15)

    is_ground &= ~veg_green
    return is_ground


def _remove_ground_from_tile(
    df_tile: pd.DataFrame,
    distance_threshold: float = 0.10,
    min_candidates: int = 100,
    max_iters: int = 10,
    min_removal_ratio: float = 0.25,
) -> pd.DataFrame:
    """Iteratively remove ground-like planes from a single tile.

    Uses a color-based prefilter to choose ground candidates, then fits a plane
    with RANSAC on candidates and removes inliers from the full tile. This is
    conceptually similar to ground_removal_only.cpp but simplified for Python.
    """
    required_cols = {"X", "Y", "Z"}
    if not required_cols.issubset(df_tile.columns):
        return df_tile

    working = df_tile.copy()
    total_removed = 0
    prev_removed: int | None = None
    it = 0

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    min_cos_angle = float(np.cos(np.deg2rad(20.0)))

    while it < max_iters and len(working) >= 100:
        mask_ground_color = _ground_like_color_mask(working)
        candidate_indices = np.nonzero(mask_ground_color)[0]

        if candidate_indices.size < min_candidates:
            break

        candidates = working.iloc[candidate_indices]
        cand_pcd = dataframe_to_o3d_point_cloud(candidates)

        try:
            plane_model, inliers_cand = cand_pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000,
            )
        except RuntimeError:
            break

        a, b, c, d = plane_model
        normal = np.array([a, b, c], dtype=np.float64)
        denom = float(np.linalg.norm(normal))
        if denom < 1e-6:
            break

        normal_unit = normal / denom
        cos_to_z = abs(float(np.dot(normal_unit, z_axis)))
        if cos_to_z < min_cos_angle:
            break

        pts = working[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
        dists = np.abs(pts @ normal + d) / denom
        inlier_mask = dists <= distance_threshold

        removed_now = int(np.count_nonzero(inlier_mask))
        if removed_now == 0:
            break

        if prev_removed is not None and removed_now < int(min_removal_ratio * prev_removed):
            break

        working = working.loc[~inlier_mask].copy()
        total_removed += removed_now
        prev_removed = removed_now
        it += 1

    return working


def remove_ground_per_tile(
    df_tiled: pd.DataFrame,
    n_tiles: int,
    distance_threshold: float = 0.10,
    min_candidates: int = 100,
    max_iters: int = 10,
    min_removal_ratio: float = 0.25,
) -> pd.DataFrame:
    """Apply ground removal to each tile separately and merge non-ground points."""
    parts: list[pd.DataFrame] = []

    for tile_id in range(n_tiles):
        sub = df_tiled[df_tiled["tile_id"] == tile_id]
        if sub.empty:
            continue

        print(f"Ground removal on tile {tile_id} with {len(sub)} points...")
        ng = _remove_ground_from_tile(
            sub,
            distance_threshold=distance_threshold,
            min_candidates=min_candidates,
            max_iters=max_iters,
            min_removal_ratio=min_removal_ratio,
        )
        print(f"  -> tile {tile_id}: {len(ng)} non-ground points")
        if not ng.empty:
            parts.append(ng)

    if not parts:
        return df_tiled.iloc[0:0].copy()

    return pd.concat(parts, ignore_index=True)


def _generate_tile_colors(n_tiles: int, seed: int = 42) -> np.ndarray:
    """Generate reproducible RGB colors in [0, 1] for each tile."""
    rng = np.random.default_rng(seed)
    colors = rng.random((n_tiles, 3)) * 0.7 + 0.3  # avoid very dark colors
    return colors


def visualize_tiles_separately(df_tiled: pd.DataFrame, n_tiles: int) -> None:
    """Show each tile in its own Open3D window, one after another."""
    colors = _generate_tile_colors(n_tiles)

    for tile_id in range(n_tiles):
        sub = df_tiled[df_tiled["tile_id"] == tile_id]
        if sub.empty:
            continue

        print(f"Tile {tile_id}: {len(sub)} points")
        pcd = dataframe_to_o3d_point_cloud(sub, color=colors[tile_id])

        o3d.visualization.draw_geometries(
            [pcd], window_name=f"LOCAL_VINEYARD_MS - Tile {tile_id}"
        )


def visualize_merged(df_tiled: pd.DataFrame, n_tiles: int) -> None:
    """Show the full point cloud with tiles colored by tile id."""
    colors = _generate_tile_colors(n_tiles)

    geoms: list[o3d.geometry.PointCloud] = []

    for tile_id in range(n_tiles):
        sub = df_tiled[df_tiled["tile_id"] == tile_id]
        if sub.empty:
            continue

        pcd = dataframe_to_o3d_point_cloud(sub, color=colors[tile_id])
        geoms.append(pcd)

    if not geoms:
        print("No points to visualize.")
        return

    o3d.visualization.draw_geometries(
        geoms, window_name="LOCAL_VINEYARD_MS - All tiles (colored by tile id)"
    )


def run_ground_removal_cpp_for_tiles(df_tiled: pd.DataFrame, n_tiles: int) -> None:
    """For each tile, export to PCD and call ground_removal_only in tile mode."""
    script_dir = os.path.dirname(__file__)
    exe_path = os.path.join(script_dir, "build/ground_removal_only")

    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"ground_removal_only executable not found at: {exe_path}")

    os.makedirs(TILES_DIR, exist_ok=True)

    for tile_id in range(n_tiles):
        sub = df_tiled[df_tiled["tile_id"] == tile_id]
        if sub.empty:
            continue

        print(f"Tile {tile_id}: exporting to PCD and running ground_removal_only...")
        pcd = dataframe_to_o3d_point_cloud(sub)
        input_pcd_path = os.path.join(TILES_DIR, f"tile_{tile_id}_input.pcd")
        o3d.io.write_point_cloud(input_pcd_path, pcd)

        subprocess.run(
            [exe_path, input_pcd_path, str(tile_id)],
            cwd=script_dir,
            check=True,
        )


def merge_tiles_from_tiles_dir(n_tiles: int) -> None:
    """Merge C++-generated per-tile nonground PCDs from TILES_DIR and save."""
    pattern = os.path.join(TILES_DIR, "tile_*_nonground.pcd")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No tile nonground PCDs found in {TILES_DIR} matching tile_*_nonground.pcd")
        return

    merged = o3d.geometry.PointCloud()

    for path in files:
        print(f"Merging: {path}")
        pcd = o3d.io.read_point_cloud(path)
        merged += pcd

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR,
        f"LOCAL_VINEYARD_MS_tiles{n_tiles}_nonground_merged.pcd",
    )
    print(f"Saving merged non-ground tiles to: {out_path}")
    o3d.io.write_point_cloud(out_path, merged)

    o3d.visualization.draw_geometries(
        [merged],
        window_name="LOCAL_VINEYARD_MS - Non-ground tiles (merged from C++ output)",
    )


def main() -> None:
    if not os.path.exists(LAS_PATH):
        raise FileNotFoundError(f"LAS file not found at: {LAS_PATH}")

    n_tiles = 20

    df = point_cloud_to_dataframe(LAS_PATH, apply_scale=True)
    df_tiled, nx, ny = assign_xy_tiles(df, n_tiles=n_tiles)

    print(
        f"Tiling result: {len(df_tiled)} points split into {n_tiles} tiles "
        f"(grid {nx} x {ny})."
    )

    # 1) Visualize each tile separately
    visualize_tiles_separately(df_tiled, n_tiles=n_tiles)

    # 2) Visualize merged point cloud colored by tile id
    visualize_merged(df_tiled, n_tiles=n_tiles)

    # 3) Ground removal per tile via C++ script, then merge and save
    run_ground_removal_cpp_for_tiles(df_tiled, n_tiles=n_tiles)
    merge_tiles_from_tiles_dir(n_tiles=n_tiles)


if __name__ == "__main__":
    main()
