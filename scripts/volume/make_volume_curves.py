#!/usr/bin/env python3
"""
make_volume_curves.py — four clean "volume vs parameter" curves.

Purpose
-------
Show how the two volume estimators behave as their single tuning parameter is
swept, on synthetic data (known truth) and on the real vineyard clusters:

  1_synthetic_voxel_curve.png   voxel volume   vs voxel size   (synthetic)
  2_synthetic_alpha_curve.png   alpha volume   vs alpha radius (synthetic)
  3_vineyard_voxel_curve.png    voxel volume   vs voxel size   (real clusters)
  4_vineyard_alpha_curve.png    alpha volume   vs alpha radius (real clusters)

Only the *volume* is plotted — deliberately NOT a paired "% error" panel. The
error curve is error(p) = (V(p) - V_true) / V_true * 100; V_true is a constant,
so error is just V(p) rescaled and shifted — an affine copy with the identical
shape. Plotting both wastes a panel. Here the truth is a horizontal reference
line instead, and the curve's distance from it *is* the error.

The estimators are imported read-only from the production pipeline, so these
curves reflect exactly what compute_row_features.py computes.

Run (needs shapely/scipy — use the app venv that already has them):
  /home/luka/Coding/DIPLOMSKI-RAD/vineyard_app/backend/.venv/bin/python \
      volume/make_volume_curves.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
for _p in (SCRIPTS / "pipeline", SCRIPTS / "synthetic"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Production estimators — the exact functions the pipeline runs.
from compute_row_features import compute_voxel_volume, compute_slice_volume  # noqa: E402
# Synthetic point samplers (same ones objct_generator.py builds scenes from).
from objct_generator import (  # noqa: E402
    sample_points_in_sphere,
    sample_points_in_cylinder,
)

# --- palette (validated dataviz default; only neutrals + 2 accents used here) -
INK = "#0b0b0b"          # primary text
SUB = "#52514e"          # secondary text / production markers
GRID = "#d8d8d2"         # recessive grid
SPREAD = "#b9b8b1"       # de-emphasised per-cluster spread lines
C_VOXEL = "#2a78d6"      # blue  — voxel method
C_ALPHA = "#1baf7a"      # aqua  — alpha-shape method
C_TRUTH = "#e34948"      # red   — ground-truth reference

# Production settings (pipeline_config.env) marked on the real-data charts.
PROD_VOXEL = 0.092
PROD_RMAX = 0.512
N_SLICES = 30

OUT_DIR = HERE / "images" / "volume_curves"


# --------------------------------------------------------------------------- #
# Synthetic scene + ground-truth union volume
# --------------------------------------------------------------------------- #
# Same object row objct_generator.py ships: cylinder–sphere–sphere–cylinder,
# packed along Y with 20% bounding-extent overlap.
SCENE_OBJECTS = [
    {"type": "cylinder", "radius": 0.35, "height": 1.80},
    {"type": "sphere",   "radius": 0.55},
    {"type": "sphere",   "radius": 0.35},
    {"type": "cylinder", "radius": 0.25, "height": 2.20},
]
OVERLAP_RATIO = 0.20
JITTER_X = 0.05
POINTS_PER_M3 = 80_000


def _y_extent(obj: dict) -> float:
    return obj["radius"] if obj["type"] == "sphere" else obj["radius"]


def build_synthetic_scene(seed: int = 123):
    """Return (points Nx3, list of shape dicts with resolved centres)."""
    rng = np.random.default_rng(seed)
    shapes, all_pts = [], []
    prev_cy = prev_ext = None
    for oid, obj in enumerate(SCENE_OBJECTS, start=1):
        ext = _y_extent(obj)
        cy = 0.0 if prev_cy is None else prev_cy + (prev_ext + ext) * (1 - OVERLAP_RATIO)
        cx = (rng.random() - 0.5) * 2 * JITTER_X
        center = (cx, cy, 0.0)
        if obj["type"] == "sphere":
            r = obj["radius"]
            n = max(5_000, int((4 / 3) * np.pi * r ** 3 * POINTS_PER_M3))
            pts = sample_points_in_sphere(n=n, radius=r, center=center, seed=1000 + oid)
            shapes.append({"type": "sphere", "c": center, "r": r})
        else:
            r, h = obj["radius"], obj["height"]
            n = max(5_000, int(np.pi * r ** 2 * h * POINTS_PER_M3))
            pts = sample_points_in_cylinder(n=n, radius=r, height=h, center=center,
                                            axis="z", seed=2000 + oid)
            shapes.append({"type": "cylinder", "c": center, "r": r, "h": h})
        all_pts.append(pts)
        prev_cy, prev_ext = cy, ext
    return np.vstack(all_pts), shapes


def monte_carlo_union_volume(shapes: list[dict], n: int = 6_000_000,
                             seed: int = 7) -> float:
    """
    Exact-in-expectation union volume: sample uniformly in the analytic bbox and
    count points inside ANY shape. Independent of the estimators being tested, so
    it is a fair ground truth (handles the overlap the analytic sum can't).
    """
    lo = np.array([np.inf, np.inf, np.inf])
    hi = -lo.copy()
    for s in shapes:
        cx, cy, cz = s["c"]
        if s["type"] == "sphere":
            r = s["r"]
            lo = np.minimum(lo, [cx - r, cy - r, cz - r])
            hi = np.maximum(hi, [cx + r, cy + r, cz + r])
        else:
            r, h = s["r"], s["h"]
            lo = np.minimum(lo, [cx - r, cy - r, cz - h / 2])
            hi = np.maximum(hi, [cx + r, cy + r, cz + h / 2])
    rng = np.random.default_rng(seed)
    p = rng.uniform(lo, hi, size=(n, 3))
    inside = np.zeros(n, dtype=bool)
    for s in shapes:
        cx, cy, cz = s["c"]
        if s["type"] == "sphere":
            inside |= ((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2
                       + (p[:, 2] - cz) ** 2) <= s["r"] ** 2
        else:
            inside |= (((p[:, 0] - cx) ** 2 + (p[:, 1] - cy) ** 2) <= s["r"] ** 2) \
                      & (np.abs(p[:, 2] - cz) <= s["h"] / 2)
    bbox_vol = float(np.prod(hi - lo))
    return float(inside.mean() * bbox_vol)


# --------------------------------------------------------------------------- #
# Sweeps
# --------------------------------------------------------------------------- #
def sweep_voxel(points, sizes):
    return np.array([compute_voxel_volume(points, float(s))["vol_voxel"] for s in sizes])


def sweep_alpha(points, radii):
    return np.array([compute_slice_volume(points, n_slices=N_SLICES, rmax=float(r))["vol_slice"]
                     for r in radii])


def convex_hull_volume(points, n_slices=N_SLICES) -> float:
    """Σ A_convex,k · dz — the rmax→∞ limit the alpha volume converges to."""
    z = points[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 0.01:
        return 0.0
    edges = np.linspace(zmin, zmax, n_slices + 1)
    total = 0.0
    for k in range(n_slices):
        lo, hi = edges[k], edges[k + 1]
        m = (z >= lo) & (z < hi) if k < n_slices - 1 else (z >= lo) & (z <= hi)
        pts = points[m]
        if len(pts) < 3:
            continue
        try:
            total += float(ConvexHull(pts[:, :2]).volume) * (hi - lo)
        except Exception:
            continue
    return total


def _crossing(x, y, target):
    """Interpolate (in log-x) the parameter where y first crosses target."""
    d = np.asarray(y) - target
    for i in range(len(d) - 1):
        if d[i] == 0:
            return float(x[i])
        if d[i] * d[i + 1] < 0:
            lx0, lx1 = np.log(x[i]), np.log(x[i + 1])
            t = -d[i] / (d[i + 1] - d[i])
            return float(np.exp(lx0 + t * (lx1 - lx0)))
    return None


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def _new_ax(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.set_xscale("log")
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="major", color="0.80", lw=0.7)
    ax.grid(True, which="minor", color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    # Readable decimal ticks across the swept range instead of bare decades.
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs="all"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    return fig, ax


def _legend(ax):
    """Standard framed legend, isolated in its own box outside the axes."""
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=True, fancybox=False, edgecolor="0.7",
              fontsize=9, borderaxespad=0.0)


def _save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / name
    # bbox_inches="tight" keeps the outside legend from being clipped.
    fig.savefig(p, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# --------------------------------------------------------------------------- #
# Synthetic figures
# --------------------------------------------------------------------------- #
def fig_synthetic_voxel(points, v_true, sizes):
    vol = sweep_voxel(points, sizes)
    fig, ax = _new_ax("Voxelization volume vs voxel size — synthetic scene",
                      "voxel size  s  [m]", "estimated volume  [m³]")
    ax.plot(sizes, vol, "-o", color=C_VOXEL, lw=2.0, ms=6,
            label="voxel estimate")
    ax.axhline(v_true, color=C_TRUTH, ls="--", lw=1.8,
               label=f"true union volume ({v_true:.2f} m³)")
    xc = _crossing(sizes, vol, v_true)
    if xc:
        ax.plot([xc], [v_true], "o", color=C_TRUTH, ms=9, zorder=5,
                label=f"estimate = truth (s ≈ {xc:.3f} m)")
    _legend(ax)
    _save(fig, "1_synthetic_voxel_curve.png")


def fig_synthetic_alpha(points, v_true, radii):
    vol = sweep_alpha(points, radii)
    convex = convex_hull_volume(points)
    fig, ax = _new_ax("Alpha-shape (slice) volume vs alpha radius — synthetic scene",
                      "alpha radius  rmax  [m]", "estimated volume  [m³]")
    ax.plot(radii, vol, "-o", color=C_ALPHA, lw=2.0, ms=6,
            label="alpha estimate")
    ax.axhline(v_true, color=C_TRUTH, ls="--", lw=1.8,
               label=f"true union volume ({v_true:.2f} m³)")
    ax.axhline(convex, color="0.35", ls=":", lw=1.4,
               label=f"convex-hull limit, rmax→∞ ({convex:.2f} m³)")
    xc = _crossing(radii, vol, v_true)
    if xc:
        ax.plot([xc], [v_true], "o", color=C_TRUTH, ms=9, zorder=5,
                label=f"estimate = truth (rmax ≈ {xc:.3f} m)")
    _legend(ax)
    _save(fig, "2_synthetic_alpha_curve.png")


# --------------------------------------------------------------------------- #
# Vineyard (real) figures — spread of clusters + mean, no absolute truth
# --------------------------------------------------------------------------- #
def _load_clusters():
    d = SCRIPTS / "out_cluster_las"
    files = sorted(d.glob("*_cluster_*_ndvi.las"))
    if not files:
        raise SystemExit(f"No cluster LAS files in {d}; run run_pipeline.sh first.")
    clouds = []
    for f in files:
        las = laspy.read(str(f))
        clouds.append(np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64))
    print(f"  loaded {len(clouds)} vineyard clusters")
    return clouds


def _spread_and_mean(ax, x, curves, color, mean_label):
    curves = np.array(curves)
    for c in curves:
        ax.plot(x, c, "-", color=SPREAD, lw=1.0, alpha=0.7, zorder=1)
    # one grey proxy handle for the legend
    ax.plot([], [], "-", color=SPREAD, lw=1.0, label="individual rows (n=%d)" % len(curves))
    ax.plot(x, curves.mean(axis=0), "-o", color=color, lw=2.6, ms=6, zorder=3,
            label=mean_label)


def fig_vineyard_voxel(clouds, sizes):
    curves = [sweep_voxel(p, sizes) for p in clouds]
    fig, ax = _new_ax("Voxelization volume vs voxel size — vineyard rows",
                      "voxel size  s  [m]", "estimated volume per row  [m³]")
    _spread_and_mean(ax, sizes, curves, C_VOXEL, "mean across rows")
    ax.axvline(PROD_VOXEL, color="0.35", ls=":", lw=1.4,
               label=f"production s = {PROD_VOXEL:g} m")
    _legend(ax)
    _save(fig, "3_vineyard_voxel_curve.png")


def fig_vineyard_alpha(clouds, radii):
    curves = [sweep_alpha(p, radii) for p in clouds]
    convex = np.mean([convex_hull_volume(p) for p in clouds])
    fig, ax = _new_ax("Alpha-shape (slice) volume vs alpha radius — vineyard rows",
                      "alpha radius  rmax  [m]", "estimated volume per row  [m³]")
    _spread_and_mean(ax, radii, curves, C_ALPHA, "mean across rows")
    ax.axhline(convex, color="0.35", ls=":", lw=1.4,
               label=f"mean convex-hull limit ({convex:.1f} m³)")
    ax.axvline(PROD_RMAX, color="0.55", ls="-.", lw=1.4,
               label=f"production rmax = {PROD_RMAX:g} m")
    _legend(ax)
    _save(fig, "4_vineyard_alpha_curve.png")


# --------------------------------------------------------------------------- #
def main():
    print("Building synthetic scene + Monte-Carlo ground truth …")
    syn_pts, shapes = build_synthetic_scene()
    v_true = monte_carlo_union_volume(shapes)
    print(f"  synthetic: {len(syn_pts):,} pts, V_true(union) = {v_true:.3f} m³")

    voxel_sizes = np.geomspace(0.01, 0.5, 24)
    alpha_radii = np.geomspace(0.02, 2.0, 24)

    print("Rendering synthetic figures:")
    fig_synthetic_voxel(syn_pts, v_true, voxel_sizes)
    fig_synthetic_alpha(syn_pts, v_true, alpha_radii)

    print("Rendering vineyard figures:")
    clouds = _load_clusters()
    fig_vineyard_voxel(clouds, voxel_sizes)
    fig_vineyard_alpha(clouds, alpha_radii)

    print(f"\nDone. 4 PNGs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
