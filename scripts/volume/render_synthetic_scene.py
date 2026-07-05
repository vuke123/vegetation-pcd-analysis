#!/usr/bin/env python3
"""
render_synthetic_scene.py — static images of the complex synthetic scene.

The synthetic validation notebook (synthetic_volume_validation_2.ipynb) builds a
deliberately awkward test scene — a row of canopy solids (spheres/boxes) with some
neighbours *touching* and some *overlapping*, small cylindrical clutter on the
soil, and a tile-wise **stepped, sloped ground** — then only shows it with an
interactive plotly widget, so no static picture ever gets saved.

This script loads that already-generated cloud from disk
(out_synth2/scene_B_stepped_ground.las, which carries the ground-truth `obj_true`
label: 1 = canopy/clutter solid, 0 = ground) and writes two PNGs:

  synthetic_scene_3d.png     one 3-D perspective (the hero view)
  synthetic_scene_views.png  side elevation (Y–Z) + top-down (X–Y)

It is read-only w.r.t. the pipeline and re-derives nothing — it just visualises
the saved scene. Run with any interpreter that has laspy + matplotlib:
  /home/luka/Coding/DIPLOMSKI-RAD/vineyard_app/backend/.venv/bin/python \
      volume/render_synthetic_scene.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SCENE_LAS = HERE / "out_synth2" / "scene_B_stepped_ground.las"
OUT_DIR = HERE / "images" / "synthetic_scene"

C_OBJ = "#1baf7a"     # canopy / clutter solids
C_GROUND = "#b8894b"  # stepped ground
INK = "#0b0b0b"
SUB = "#52514e"


def load_scene(path: Path):
    if not path.is_file():
        raise SystemExit(f"Scene LAS not found: {path}\n"
                         "Run synthetic_volume_validation_2.ipynb first to generate it.")
    las = laspy.read(str(path))
    xyz = np.stack([las.x, las.y, las.z], axis=-1).astype(np.float64)
    obj = np.asarray(las.obj_true).astype(int) if "obj_true" in las.point_format.dimension_names \
        else (np.asarray(las.classification) != 2).astype(int)
    return xyz, obj


def _equal_box_aspect(ax, xyz):
    """Give the 3-D axes true 1:1:1 data proportions (no distortion)."""
    ranges = xyz.max(axis=0) - xyz.min(axis=0)
    ax.set_box_aspect(tuple(ranges))


def render_3d(xyz, obj, out_png):
    g, o = obj == 0, obj == 1
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[g, 0], xyz[g, 1], xyz[g, 2], s=2, c=C_GROUND,
               alpha=0.35, linewidths=0, depthshade=True)
    ax.scatter(xyz[o, 0], xyz[o, 1], xyz[o, 2], s=3, c=C_OBJ,
               alpha=0.75, linewidths=0, depthshade=True)
    _equal_box_aspect(ax, xyz)
    ax.view_init(elev=20, azim=-72)
    ax.set_xlabel("X [m]", color=SUB)
    ax.set_ylabel("Y [m]  (smjer reda)", color=SUB)
    ax.set_zlabel("Z [m]", color=SUB)
    ax.set_title("Složena sintetička scena: red krošnji i smetnje na stepenastom tlu",
                 fontsize=13, color=INK, pad=6)
    handles = [
        Line2D([0], [0], marker="o", ls="", color=C_OBJ, ms=8,
               label=f"objekti (krošnja + smetnje)  (n = {int(o.sum()):,})"),
        Line2D([0], [0], marker="o", ls="", color=C_GROUND, ms=8,
               label=f"stepenasto tlo  (n = {int(g.sum()):,})"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, fontsize=9,
              edgecolor="0.7")
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def render_views(xyz, obj, out_png):
    g, o = obj == 0, obj == 1
    fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(14, 6))

    # Side elevation (Y–Z): shows the stepped/sloped ground and canopy bumps.
    ax_side.scatter(xyz[g, 1], xyz[g, 2], s=2, c=C_GROUND, alpha=0.4, linewidths=0)
    ax_side.scatter(xyz[o, 1], xyz[o, 2], s=3, c=C_OBJ, alpha=0.7, linewidths=0)
    ax_side.set_title("Bočni prikaz (Y–Z)", fontsize=12, color=INK)
    ax_side.set_xlabel("Y [m]  (smjer reda)", color=SUB)
    ax_side.set_ylabel("Z [m]", color=SUB)
    ax_side.set_aspect("equal")
    ax_side.grid(True, color="0.85", lw=0.6)
    ax_side.set_axisbelow(True)

    # Top-down (X–Y): shows the row layout and touching/overlapping units.
    ax_top.scatter(xyz[g, 0], xyz[g, 1], s=2, c=C_GROUND, alpha=0.4, linewidths=0)
    ax_top.scatter(xyz[o, 0], xyz[o, 1], s=3, c=C_OBJ, alpha=0.7, linewidths=0)
    ax_top.set_title("Tlocrt (X–Y)", fontsize=12, color=INK)
    ax_top.set_xlabel("X [m]", color=SUB)
    ax_top.set_ylabel("Y [m]  (smjer reda)", color=SUB)
    ax_top.set_aspect("equal")
    ax_top.grid(True, color="0.85", lw=0.6)
    ax_top.set_axisbelow(True)

    handles = [
        Line2D([0], [0], marker="o", ls="", color=C_OBJ, ms=8,
               label="objekti (krošnja + smetnje)"),
        Line2D([0], [0], marker="o", ls="", color=C_GROUND, ms=8,
               label="stepenasto tlo"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True,
               fontsize=10, edgecolor="0.7", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Složena sintetička scena — ortografski prikazi",
                 fontsize=13, color=INK, y=1.07)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}")


def main():
    xyz, obj = load_scene(SCENE_LAS)
    print(f"Loaded {len(xyz):,} points from {SCENE_LAS.name} "
          f"({int((obj == 1).sum()):,} object / {int((obj == 0).sum()):,} ground)")
    render_3d(xyz, obj, OUT_DIR / "synthetic_scene_3d.png")
    render_views(xyz, obj, OUT_DIR / "synthetic_scene_views.png")
    print(f"\nDone. PNGs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
