# Pipeline Contracts

This document defines the data contracts between pipeline jobs.

## Folder Structure Convention

```
bronze/
  vineyard_id=<vineyard_id>/
    flight_date=YYYY-MM-DD/
      flight_id=<flight_id>/
        sensors/
          ms/input.las
          ir/input.las (optional)
          lidar/input.las (optional)

silver/
  run_id=<run_id>/
    ms/
      input.laz          # normalized from bronze (LAS→LAZ)
      non_ground.laz     # output of job1_ground_removal
      ground.laz         # optional
    rows/
      row_index.parquet  # output of job2_clustering
    clusters/
      cluster_00.laz
      cluster_01.laz
      ...
    stage=ground/
      _SUCCESS.json
    stage=cluster/
      _SUCCESS.json

gold/
  run_id=<run_id>/
    features/
      features_per_row.parquet  # output of job3_features
      qc_summary.json
    stage=features/
      _SUCCESS.json
```

## run_id Convention

Format: `YYYYMMDD_HHMMSS_<8-char-uuid>`

Example: `20260127_230000_a1b2c3d4`

---

## Contract 1: row_index.parquet (Output of Job 2)

This file is the contract between clustering and feature extraction.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| run_id | string | No | Run identifier |
| row_id | int32 | No | Row index (0-based) |
| point_count | int64 | No | Number of points in this row |
| bbox_minx | float64 | No | Bounding box min X |
| bbox_miny | float64 | No | Bounding box min Y |
| bbox_minz | float64 | No | Bounding box min Z |
| bbox_maxx | float64 | No | Bounding box max X |
| bbox_maxy | float64 | No | Bounding box max Y |
| bbox_maxz | float64 | No | Bounding box max Z |
| crs_epsg | int32 | Yes | EPSG code if known |
| crs_wkt | string | Yes | WKT string if EPSG unknown |
| centerline_wkt | string | Yes | Optional centerline geometry |
| poly_wkt | string | Yes | Optional polygon geometry |
| cluster_file | string | Yes | Path to cluster LAZ file |

---

## Contract 2: features_per_row.parquet (Output of Job 3)

Final output with per-row features.

| Field | Type | Nullable | Description |
|-------|------|----------|-------------|
| **Keys** |
| run_id | string | No | Run identifier |
| vineyard_id | string | No | Vineyard identifier |
| flight_id | string | No | Flight identifier |
| row_id | int32 | No | Row index |
| **NDVI Stats** |
| ndvi_mean | float64 | No | Mean NDVI |
| ndvi_std | float64 | No | Std dev of NDVI |
| ndvi_p10 | float64 | No | 10th percentile NDVI |
| ndvi_p90 | float64 | No | 90th percentile NDVI |
| ndvi_low_frac | float64 | No | Fraction with NDVI < threshold |
| **IR Stats (nullable)** |
| temp_mean | float64 | Yes | Mean temperature |
| temp_max | float64 | Yes | Max temperature |
| temp_std | float64 | Yes | Std dev temperature |
| **Volume (nullable)** |
| vol_voxel | float64 | Yes | Voxel-based volume |
| vol_slicing | float64 | Yes | Slicing-based volume |
| vol_hull | float64 | Yes | Convex hull volume |
| vol_disagreement | float64 | Yes | Disagreement between methods |
| **QC** |
| seg_confidence | float64 | Yes | Segmentation confidence |
| gap_score | float64 | Yes | Gap detection score |
| warnings | string | Yes | JSON-encoded warnings list |

---

## Contract 3: _SUCCESS.json (Success Markers)

Written at the end of each job to signal completion.

```json
{
  "stage": "ground|cluster|features",
  "run_id": "<run_id>",
  "timestamp": "2026-01-27T23:00:00",
  "input_files": ["path/to/input1", "path/to/input2"],
  "output_files": ["path/to/output1"],
  "metrics": {
    "duration_s": 123.45,
    "point_count": 1000000
  },
  "errors": []
}
```

---

## Data Sources (Allowed)

Only these files under `/datasource/flights/` may be used:

- `07-15-MS.laz` - Multispectral data (July 15)
- `08-19-MS.laz` - Multispectral data (August 19)
- `07-15-LIDAR.laz` - LiDAR data (July 15)
- `2025-07-15-IR.laz` - Infrared data (July 15)
