# Canopy structure and per-row features — methodology report

This report explains **why** and **how** the pipeline driven by
`scripts/run_pipeline.sh` computes per-row canopy features. It documents
the two feature-extraction stages that come at the end of the pipeline:

1. `compute_canopy_structure.py` — **canopy structure** (porosity, gap
   fraction, LAI proxy, LAD proxy) using a segment-by-segment voxel
   analysis along each row.
2. `compute_row_features.py` — **geometric & radiometric** row features
   (height, length/width/azimuth, volume, NDVI statistics).

The focus, as requested, is on **LAI, gap fraction, and porosity**: what
they mean, why we compute them this way, and how the algorithm gets from
a point cloud to a number.

> All metrics here are **structural proxies** derived from a discrete-return
> multispectral point cloud (drone, ~July 2025). They are not equivalent
> to field-measured LAI or physically modelled LAD. They are valid for
> *relative* comparisons across rows within the same acquisition.

---

## 0. Where these features sit in the pipeline

`run_pipeline.sh` runs six stages, and the feature stage is the last one:

```
0) clean output dirs
1) SMRF ground classification  → out_ground/*_nonground.las
2) build C++ targets (cmake)
3) Euclidean clustering (C++)  → out_cluster/cluster_NN.pcd
4) per-cluster NDVI LAS        → out_cluster_las/cluster_NN_ndvi.las
5) PDAL merge into one LAS
6) compute per-row features    → row_features.parquet
```

Steps 1–5 deliver, for each row, a self-contained LAS file containing
**only above-ground vegetation points** for that row, with NDVI attached
per point. The feature stage assumes that input.

That assumption is the whole reason the metrics in step 6 can be computed
locally per row instead of having to re-segment vegetation each time.

---

## 1. Why canopy structure at all?

In precision viticulture you need numbers that describe how dense or how
open a vine row is, separately from the row's NDVI. Two rows can have
identical mean NDVI but very different yields, vigour status and pruning
needs because:

- One is a continuous, dense canopy wall (low gap fraction, high LAI).
- The other has the same per-leaf NDVI but visible holes between vines
  (high gap fraction, low LAI), so total intercepted light per metre of
  row is much lower.

Canopy **structure** captures the second axis. NDVI alone cannot, because
NDVI saturates and reflects per-leaf physiology, not how much canopy is
actually there.

The three structural metrics the pipeline reports are:

| Metric | What it measures | Physical analogue |
|---|---|---|
| **Porosity** | Fraction of empty voxels inside the row's bounding volume | "Airiness" of the row in 3D |
| **Gap fraction** | Fraction of empty cells in the cross-row (v–z) plane | What a side-looking camera would see through the canopy wall |
| **LAI (proxy)** | Beer–Lambert inversion of gap fraction | Leaf area per unit ground area |
| **LAD (proxy)** | Vertical profile of layer-wise LAI density | Leaf area density at each height |

All four come from the **same** discrete voxelization. We just summarise
the occupancy grid in three different ways.

---

## 2. The segmentation-along-the-row trick (why we do it locally)

This is the single most important methodological decision in
`compute_canopy_structure.py` and worth understanding before any formula.

### The problem with whole-row voxelization

A vine row in this dataset is typically 20–110 m long, ~1.0–1.5 m wide,
~1.5–2.5 m tall. If you voxelize the entire row's axis-aligned bounding
box at, say, 10 cm voxels, you get on the order of:

```
N_voxels ≈ (100 / 0.10) × (1.5 / 0.10) × (3.0 / 0.10)
         ≈ 1000 × 15 × 30 ≈ 4.5 × 10⁵
```

…but the canopy is a thin meandering ribbon inside that box. Most voxels
sit in air to the left, right, top, or above gaps along the row. Porosity
computed on that grid will be ~0.99 and gap fraction ~0.95 — values that
say "this box has a lot of air in it", not "this canopy is open". LAI
derived from such a gap fraction is meaningless.

### The fix: local segment analysis

Instead, the pipeline:

1. **Rotates each row** into a local frame where `u` is along the row
   and `v` is across it (PCA on the XY coordinates picks `u`).
2. **Splits the row** along `u` into segments of `segment_length` metres
   (default 1.0 m).
3. **Voxelizes each segment in its own tight bounding box.** A 1 m
   segment containing canopy is ~1.0 × 1.0 × 2.0 m, so the surrounding
   air is minimal.
4. **Aggregates** the per-segment metrics into row-level mean / p50 / p90.

This produces structure metrics that reflect the canopy's internal
openness, not the emptiness of a 100 m long box around it.

The rotation uses simple PCA on XY:

```
centred = xy − mean(xy)
cov     = cov(centred)
u_dir   = eigenvector of cov with the largest eigenvalue
v_dir   = u_dir rotated 90° in the plane
```

implemented in `_estimate_row_direction` and `_project_to_local`.

---

## 3. Porosity

### What it is

Porosity is the fraction of voxels inside the segment's 3D bounding box
that contain **no points**:

```
porosity = 1 − n_filled / n_total
```

where:

- `n_total = nu × nv × nz` — the number of voxels in the segment grid,
- `n_filled` = number of unique voxel indices that hold ≥1 point.

### Why it is calculated this way

A point cloud is a discrete sample of surfaces. We can't directly observe
"is this cubic metre of air full of leaves?", but we can answer "did the
sensor ever return a point from this small cubic cell?". The voxel grid
quantises that question:

- A cell with at least one return is treated as occupied.
- A cell with no returns is treated as empty (air or fully occluded).

Porosity, computed on the **segment's own bounding box** rather than the
full row's, then approximates *internal airiness of the canopy cross-section*
in that segment.

### Per-row aggregation

`_aggregate_segments` reports `porosity_mean`, `porosity_p50`, and
`porosity_p90` across the valid segments of the row. The p90 is useful
for spotting rows with isolated heavy-foliage sections.

### Sensitivity caveats

Porosity depends strongly on the voxel size:

- Too small (e.g. 2 cm at this point density): most voxels touch ≤1
  point, porosity is artificially high and noisy.
- Too large (e.g. 30 cm): every voxel is occupied, porosity collapses to
  near zero and the metric loses discriminative power.

The default in the pipeline is **10 cm**. The `lai_voxel_size_sensitivity.py`
script (deliverable 3) lets you see this dependency empirically.

---

## 4. Gap fraction

### What it is

Gap fraction is the fraction of *cross-row* (v–z plane) cells that
contain no canopy:

```
gap_fraction = 1 − n_occupied_vz / n_total_vz
```

where:

- We collapse the segment's voxel grid along `u` (along-row).
- A (v, z) cell is **occupied** if *any* u-voxel at that (v, z) is filled.
- `n_total_vz = nv × nz`.

### Why the v–z plane (not top-down)

For vineyard rows, the canopy is functionally a **wall**, not a closed
overstory. The biologically meaningful gap is what you see when you
project the canopy onto the cross-row vertical plane — the view a
sun-tracking light meter or a side-looking sensor would integrate.

Top-down gap fraction (projecting onto the horizontal plane) is the
usual forestry definition. For a continuous forest canopy this is
correct. For a 1 m wide trellised vine row inside a 3 m row spacing,
top-down gap fraction is dominated by inter-row soil and reads ~0.95
regardless of vine health. The v–z formulation gives a much better
signal for vine canopies.

### Why `unique_voxels[:, 1:3]` and a set

In code:

```python
vz_columns = set(map(tuple, unique_voxels[:, 1:3]))
gap_fraction = 1.0 - len(vz_columns) / (nv * nz)
```

Indices 1 and 2 are `v` and `z`. Putting them in a `set` deduplicates
along `u` in one line — that's the "collapse onto the v–z plane" step.

### Aggregation

Per-row outputs are `gap_fraction_mean`, `gap_fraction_p50`,
`gap_fraction_p90`. The mean across segments gives the row-typical wall
openness; the p90 picks out the segment(s) with the biggest hole in the
row, which is exactly what you want for spotting missing vines or
freshly-pruned gaps.

---

## 5. LAI (proxy) — Beer–Lambert inversion

### Where the formula comes from

The Beer–Lambert law for canopy radiative transfer states that the
probability that a beam passes through a canopy of leaf area index `L`
without being intercepted is:

```
P(gap) = exp( − G · L / cos(θ) )
```

- `L`  — leaf area index (one-sided leaf area per unit ground area).
- `G`  — projection of unit leaf area in the direction of the beam.
  For a **spherical** (random) leaf angle distribution viewed from nadir,
  `G = 0.5`.
- `θ`  — zenith angle of the beam. For nadir, `cos(θ) = 1`.

Solving for `L`:

```
L = − ln(P(gap)) / G
```

This is what the code does in `_compute_segment_metrics`:

```python
p_gap     = max(gap_fraction, EPS)
lai_proxy = -np.log(p_gap) / G_FUNCTION    # G_FUNCTION = 0.5
```

`EPS = 1e-6` is the numeric guard against `log(0)` for segments where
every cross-row cell is occupied.

### Why this is a proxy, not "real LAI"

Three honest limitations to remember:

1. **Discrete-return cloud, not transmittance.** True Beer–Lambert needs
   actual radiation transmission probability. We substitute *geometric*
   gap fraction from a voxel grid. They are correlated but not identical.
2. **`G = 0.5` is an assumption.** Vine canopies usually have a more
   horizontal leaf distribution; a more accurate `G` would be ~0.4–0.45
   for plagiophile distributions. Using 0.5 keeps the value comparable
   to standard literature defaults at the cost of a small systematic
   bias.
3. **Nadir-equivalent geometry.** The cross-row v–z plane corresponds
   to a horizontal view of the canopy wall, not a nadir-down view. We
   are using Beer–Lambert with a `G` calibrated for nadir, applied to a
   horizontal projection. This is conventional for vineyard "vertical
   canopy LAI" but should be flagged as such in the thesis.

The LAI numbers this produces are therefore best reported as
`lai_proxy_*` (the column names in the parquet already do this) and
discussed as a relative structural index.

### Per-row aggregation

Reported as `lai_proxy_mean`, `lai_proxy_p50`, `lai_proxy_p90`. The mean
across segments is the row-level LAI proxy; the p90 isolates the densest
sections.

---

## 6. LAD (proxy) — vertical profile

### What it is

Where LAI summarises the whole segment, LAD (leaf area density) tells
you **where in the canopy the leaves are**: how much canopy is in the
0.5–1.0 m band vs the 1.0–1.5 m band, etc. This matters for spray
targeting and pruning decisions.

### Formula

For each vertical layer `k` of height `dz` (= one voxel):

```
P_gap_layer(k) = 1 − (n_v_cells_filled_in_layer / nv)
LAD(k)         = − ln( max(P_gap_layer(k), EPS) ) / ( G · dz )
```

i.e. the same Beer–Lambert inversion, but applied per height slice using
the gap fraction across `v` only. We average over layers that have
*some* canopy in them (skipping fully-empty layers above/below the row),
provided we have at least `MIN_LAYERS_FOR_LAD = 3` valid layers.

```python
for k in range(nz):
    layer_voxels = unique_voxels[unique_voxels[:, 2] == k]
    if len(layer_voxels) == 0:        # empty layer → skip
        continue
    layer_v = set(layer_voxels[:, 1].tolist())
    p_gap_layer = 1.0 - len(layer_v) / nv
    if p_gap_layer >= 1.0 - EPS:      # near-empty → skip
        continue
    lad_layer = -np.log(max(p_gap_layer, EPS)) / (G_FUNCTION * dz)
    lad_values.append(lad_layer)
lad_mean = float(np.mean(lad_values)) if len(lad_values) >= MIN_LAYERS_FOR_LAD else nan
```

The reported `lad_proxy_mean` is the average leaf area density across
populated height layers of the segment.

---

## 7. Per-row features in `compute_row_features.py`

This script handles features that are *not* canopy density. They are
computed once over the full row (not per-segment), because they describe
the row's overall shape and radiometry.

### 7.1 Slope-aware height

The cluster LAS is above-ground vegetation but is **not** referenced to a
true ground surface — heights are absolute Z coordinates and the
underlying ground may slope. To get a sensible "canopy height above
local ground" without rerunning SMRF, the script fits a ground plane
**locally**:

1. Take the lowest 10 % of points (by Z) as candidate ground.
2. RANSAC: repeatedly pick 3 random points, fit a plane, count inliers
   within `dist_thresh = 0.10 m`. Reject near-vertical planes
   (`n_z < 0.707`, i.e. > 45° slope).
3. Use the best-inlier plane as the local ground. Project every point's
   signed distance to that plane as its height.
4. Shift heights so the 2nd-percentile height is 0 (robust against
   ground-misclassified noise).

Outputs: `height_max`, `height_mean`, `height_std`, `height_cv`,
`height_p50`, `height_p90`, plus the ground plane's slope in degrees
(`ground_slope_deg`).

The CV `(σ_h / μ_h)` is a single-number canopy uniformity index:
low CV = tidy hedge, high CV = ragged canopy.

### 7.2 Row geometry (PCA)

The same PCA used in canopy structure also gives row geometry features
in one shot:

- `row_length` = extent along principal axis.
- `row_width` = extent perpendicular to it.
- `azimuth_deg` = compass bearing of `u_dir` from north, clockwise. Used
  for sun-exposure analysis.

### 7.3 Volume — two estimators

The pipeline reports two volume estimates **on purpose**, because each
is biased in a different direction:

#### Voxel volume (`vol_voxel`)

```
vol_voxel = n_unique_voxels × voxel_size³
```

A simple occupancy-grid volume. Default voxel size is adaptive:
`min(bounds) × 0.05`, clamped to `[0.01, 0.20]` m. For a 1.2 m wide row
this lands around 6 cm. This estimator:

- *Underestimates* solid volume because it skips voxels with no return
  (shaded interior of the canopy is "empty" to the sensor).
- *Overestimates* foliage volume because it counts the air gaps between
  leaves as canopy.

#### Slice + alpha-hull volume (`vol_slice`)

For 30 horizontal Z-slices:
1. Take the slice's XY points.
2. Triangulate them (Delaunay).
3. Keep only triangles with circumradius `R ≤ rmax` — that's the
   α-shape (concave hull) filter.
4. Polygonize the boundary, sum polygon area `A_k`.
5. `vol_slice = Σ_k A_k · dz`.

`rmax` is auto-estimated from the median nearest-neighbour distance in
XY (`rmax = max(10 × nn_dist, 0.5)`). This estimator behaves more like
a "wrapped solid" — it captures the canopy envelope but doesn't punish
internal sparsity, so it tends to be larger than `vol_voxel` for the
same row.

Reporting both lets you check both directions of bias and lets the
thesis pick whichever is more appropriate for the analysis (e.g. yield
correlation usually prefers `vol_voxel` because it tracks actual foliage
material; pruning-volume analysis prefers `vol_slice`).

### 7.4 NDVI statistics

Per cluster:

- `ndvi_mean`, `ndvi_std`
- `ndvi_p10`, `ndvi_p90`, `ndvi_range = p90 − p10`
- `ndvi_low_frac` = share of points with NDVI < 0.2 (a coarse senescent /
  stress indicator).

These are the **inputs** to the species-aware NDVI comparison
(`scripts/compare_ndvi_species.py`) and the basis for any vigour map.

---

## 8. Putting it together — what a final row record looks like

After `compute_row_features.py` and `compute_canopy_structure.py` you
have, per row:

```
row_features.parquet
  identity           : cluster_file, row_id, point_count, points_per_m
  geometry           : row_length, row_width, azimuth_deg, centroid_*, bbox_*
  height             : height_{max, mean, std, cv, p50, p90}, ground_slope_deg
  volume             : vol_voxel, n_voxels, vol_voxel_per_m, vol_slice, n_slices_used
  NDVI               : ndvi_{mean, std, p10, p90, low_frac, range}

row_canopy_structure.parquet
  identity / params  : cluster_file, row_id, point_count, segment_length, voxel_size
  segments           : n_segments_total, n_segments_valid
  structure (×4)     : {porosity, gap_fraction, lai_proxy, lad_proxy}_{mean, p50, p90}
  bbox               : bbox_{x, y, z}_extent, height_range, bbox_minz/maxz
```

Together these two tables give a full per-row description: where the row
is (`centroid_*`, `azimuth_deg`), how big it is (`row_length`,
`row_width`, `height_*`, volumes), how dense it is (`porosity`,
`gap_fraction`, `lai_proxy`, `lad_proxy`), and how vigorous it is
(`ndvi_*`). That is everything the thesis needs as a feature table for
correlation with yield/quality or for clustering rows by management
need.

---

## 9. Quick reference — parameters and their effect

| Parameter | Default | Where | Effect |
|---|---|---|---|
| `segment_length` | 1.0 m | `compute_canopy_structure.py` | Smaller = more local detail, noisier per-segment stats; larger = smoother but less localized |
| `voxel_size` | 0.10 m | `compute_canopy_structure.py` | See sensitivity script; too small → noisy porosity, too large → saturated grid |
| `MIN_POINTS_PER_SEGMENT` | 10 | `compute_canopy_structure.py` | Skips segments with too few points |
| `G_FUNCTION` | 0.5 | `compute_canopy_structure.py` | Leaf-angle factor in LAI inversion; spherical default |
| `MIN_LAYERS_FOR_LAD` | 3 | `compute_canopy_structure.py` | Floor on layers before LAD is reported |
| `low_fraction` | 0.10 | `compute_row_features.py` ground RANSAC | Bottom Z fraction used as ground candidates |
| `dist_thresh` | 0.10 m | `compute_row_features.py` ground RANSAC | Inlier band for ground plane |
| `voxel_size` (volume) | auto, ∈ [0.01, 0.20] | `compute_row_features.py` | Voxel volume granularity |
| `n_slices` | 30 | `compute_row_features.py` slice volume | Z-resolution of the α-hull stack |
| `rmax` (α-shape) | `max(10·nn_dist, 0.5)` | `compute_row_features.py` slice volume | Concavity of α-hull boundary; larger = more convex |

---

## 10. Suggested validation work for the thesis

These metrics are proxies; they are only as good as their calibration.
Items to consider:

1. **Voxel-size sensitivity.** Run `lai_voxel_size_sensitivity.py` for a
   representative set of rows and report the curve. Discuss whether the
   chosen default (10 cm) is in a stable plateau or on a steep slope.
2. **Segment-length sensitivity.** Same idea, but vary `--segment-length`
   between 0.25 m and 5.0 m. Confirm that means/p50 stabilise at the
   chosen default and that the cross-row width is fully captured.
3. **`G` ablation.** Recompute LAI proxy with `G ∈ {0.4, 0.5, 0.6}`.
   Report how much the relative ranking of rows changes (you should
   find: very little — that is the value of using these as relative
   indices).
4. **Cross-species reference windows.** The NDVI comparison script uses
   default windows (olive 0.55–0.85, vineyard 0.35–0.70). Calibrate
   these from your dataset before drawing thesis conclusions.
