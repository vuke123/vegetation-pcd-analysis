import open3d as o3d
import numpy as np
import glob
import os
import csv


OUT_DIR = "out_ground"
CLUSTER_DIR = "out_cluster"


def load_pcd(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    pcd = o3d.io.read_point_cloud(filepath)
    print(f"Loaded {filepath}: {len(pcd.points)} points")
    return pcd


def choose_file(pattern, title="Choose a file"):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files match: {pattern}")
        return None
    if len(files) == 1:
        return files[0]

    print(f"\n{title}")
    for i, f in enumerate(files):
        print(f"  {i+1}. {os.path.basename(f)}")
    sel = input("Enter number: ").strip()
    if not sel.isdigit():
        return None
    idx = int(sel) - 1
    if 0 <= idx < len(files):
        return files[idx]
    return None


def load_ground_removal_stats(nonground_path):
    log_path = os.path.join(OUT_DIR, "ground_removal_log.csv")
    if not os.path.exists(log_path):
        return None

    target = os.path.normpath(nonground_path)

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_path = os.path.normpath(row.get("nonground_file", ""))
            if row_path == target:
                try:
                    orig_pts = int(row["orig_pts"])
                    voxel_pts = int(row["voxel_pts"])
                    total_removed = int(row["total_removed"])
                    final_nonground = int(row["final_nonground"])
                except (KeyError, ValueError):
                    return None
                return {
                    "orig_pts": orig_pts,
                    "voxel_pts": voxel_pts,
                    "total_removed": total_removed,
                    "final_nonground": final_nonground,
                }

    return None


def visualize_original():
    print("\n" + "="*60)
    print("1. ORIGINAL POINT CLOUD")
    print("="*60)

    path = os.path.join(OUT_DIR, "original_cloud.pcd")
    pcd = load_pcd(path)
    if pcd is None:
        return
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd], window_name="Original Point Cloud")


def visualize_downsampled():
    print("\n" + "="*60)
    print("2. DOWNSAMPLED POINT CLOUD (per leaf size)")
    print("="*60)

    pattern = os.path.join(OUT_DIR, "downsampled_leaf*cm.pcd")
    filepath = choose_file(pattern, title="Pick a downsampled cloud")
    if not filepath:
        return

    pcd = load_pcd(filepath)
    if pcd is None:
        return
    pcd.paint_uniform_color([0.3, 0.6, 0.9])
    o3d.visualization.draw_geometries([pcd], window_name="Downsampled Point Cloud")


def visualize_nonground():
    print("\n" + "="*60)
    print("3. NON-GROUND POINT CLOUD (after RANSAC ground removal)")
    print("="*60)

    pattern = os.path.join(OUT_DIR, "nonground_leaf*cm_dist*cm*.pcd")
    filepath = choose_file(pattern, title="Pick a non-ground cloud (leaf/dist)")
    if not filepath:
        return

    pcd = load_pcd(filepath)
    if pcd is None:
        return

    stats = load_ground_removal_stats(filepath)
    window_title = "Non-Ground (after RANSAC)"
    if stats:
        orig = stats["orig_pts"]
        removed = stats["total_removed"]
        remaining = stats["final_nonground"]
        ratio = (removed / orig * 100.0) if orig > 0 else 0.0
        print(
            f"Ground removal stats: orig={orig}, removed={removed}, "
            f"remaining={remaining} ({ratio:.1f}% removed)"
        )
        window_title = (
            f"Non-Ground (removed {removed}/{orig} pts, "
            f"{ratio:.1f}% of original)"
        )

    pcd.paint_uniform_color([0.2, 0.8, 0.4])
    o3d.visualization.draw_geometries([pcd], window_name=window_title)


def visualize_clusters():
    print("\n" + "="*60)
    print("4. CLUSTERS (by config id)")
    print("="*60)

    config = input("Enter config id (integer): ").strip()
    if not config.isdigit():
        print("Invalid config id.")
        return

    pattern = os.path.join(CLUSTER_DIR, f"config{config}_leaf*.pcd")
    cluster_files = sorted(glob.glob(pattern))
    if not cluster_files:
        print(f"No cluster files found for config {config}. Expected: {pattern}")
        return

    np.random.seed(42)
    colors = np.random.rand(len(cluster_files), 3) * 0.7 + 0.3

    geoms = []
    for i, fp in enumerate(cluster_files):
        pcd = load_pcd(fp)
        if pcd:
            pcd.paint_uniform_color(colors[i])
            geoms.append(pcd)

    if geoms:
        o3d.visualization.draw_geometries(geoms, window_name=f"Clusters config {config}")


def visualize_comparison():
    print("\n" + "="*60)
    print("5. COMPARISON (Original vs Non-ground vs Clusters)")
    print("="*60)

    config = input("Enter config id for clusters (optional, Enter to skip clusters): ").strip()

    geoms = []

    original = load_pcd(os.path.join(OUT_DIR, "original_cloud.pcd"))
    if original:
        original.paint_uniform_color([0.5, 0.5, 0.5])
        bbox = original.get_axis_aligned_bounding_box()
        width = bbox.max_bound[0] - bbox.min_bound[0]
        geoms.append(original)
    else:
        width = 5.0

    # Non-ground: let user pick a leaf/dist file
    ng_file = choose_file(os.path.join(OUT_DIR, "nonground_leaf*cm_dist*cm*.pcd"),
                          title="Pick non-ground cloud for comparison")
    stats = None
    if ng_file:
        ng = load_pcd(ng_file)
        if ng:
            ng.paint_uniform_color([0.2, 0.8, 0.4])
            ng.translate([width * 1.2, 0, 0])
            geoms.append(ng)
            stats = load_ground_removal_stats(ng_file)

    # Clusters: optional
    if config.isdigit():
        cluster_files = sorted(
            glob.glob(os.path.join(CLUSTER_DIR, f"config{config}_leaf*.pcd"))
        )
        if cluster_files:
            np.random.seed(42)
            colors = np.random.rand(len(cluster_files), 3) * 0.7 + 0.3
            for i, fp in enumerate(cluster_files):
                pcd = load_pcd(fp)
                if pcd:
                    pcd.paint_uniform_color(colors[i])
                    pcd.translate([width * 2.4, 0, 0])
                    geoms.append(pcd)

    if geoms:
        window_title = "Comparison"
        if stats:
            orig = stats["orig_pts"]
            removed = stats["total_removed"]
            remaining = stats["final_nonground"]
            ratio = (removed / orig * 100.0) if orig > 0 else 0.0
            print(
                f"Ground removal stats: orig={orig}, removed={removed}, "
                f"remaining={remaining} ({ratio:.1f}% removed)"
            )
            window_title = (
                f"Comparison (removed {removed}/{orig} pts, "
                f"{ratio:.1f}% of original)"
            )

        print("Left: Original | Middle: Non-ground | Right: Clusters (optional)")
        o3d.visualization.draw_geometries(geoms, window_name=window_title)

def main():
    print("="*60)
    print("PCL Clustering Visualization")
    print(f"Using ground output directory: {OUT_DIR}")
    print(f"Using cluster output directory: {CLUSTER_DIR}")
    print("="*60)

    while True:
        print("\nOptions:")
        print("  1. View original point cloud")
        print("  2. View downsampled point cloud")
        print("  3. View non-ground (after RANSAC ground removal)")
        print("  4. View clusters (by config id)")
        print("  5. View comparison")
        print("  q. Quit")

        choice = input("\nEnter choice: ").strip().lower()

        if choice == "1":
            visualize_original()
        elif choice == "2":
            visualize_downsampled()
        elif choice == "3":
            visualize_nonground()
        elif choice == "4":
            visualize_clusters()
        elif choice == "5":
            visualize_comparison()
        elif choice == "q":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
