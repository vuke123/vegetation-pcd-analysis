# Azure Platform Pipeline

This folder contains the refactored vineyard point cloud analysis pipeline, split into 3 independent jobs for Azure migration.

## Quick Start (Local)

```bash
# Make scripts executable
chmod +x *.sh

# Run full pipeline locally
./run_all.sh --input ../datasource/flights/07-15-MS.laz

# Or run jobs individually
./job1_ground_removal.sh --input ../datasource/flights/07-15-MS.laz --run-id my_run_001
./job2_clustering.sh --run-id my_run_001
./job3_features.sh --run-id my_run_001
```

## Docker Usage

### Build Image

```bash
# From project root (DIPLOMSKI-RAD/)
docker build -t vineyard-pipeline:dev -f azure_platform/Dockerfile .
```

### Run Jobs in Container

```bash
# Job 1: Ground Removal
docker run --rm \
  -v "$(pwd)":/data \
  -e RUN_ID="test_001" \
  -e INPUT_MS="/data/datasource/flights/07-15-MS.laz" \
  -e OUT_BASE="/data" \
  vineyard-pipeline:dev \
  bash /app/azure_platform/job1_ground_removal.sh

# Job 2: Clustering
docker run --rm \
  -v "$(pwd)":/data \
  -e RUN_ID="test_001" \
  -e OUT_BASE="/data" \
  vineyard-pipeline:dev \
  bash /app/azure_platform/job2_clustering.sh

# Job 3: Features
docker run --rm \
  -v "$(pwd)":/data \
  -e RUN_ID="test_001" \
  -e OUT_BASE="/data" \
  -e VINEYARD_ID="vinograd_1" \
  -e FLIGHT_ID="2025-07-15" \
  vineyard-pipeline:dev \
  bash /app/azure_platform/job3_features.sh

# Or run all jobs in sequence
docker run --rm \
  -v "$(pwd)":/data \
  -e RUN_ID="test_001" \
  -e INPUT_MS="/data/datasource/flights/07-15-MS.laz" \
  -e OUT_BASE="/data" \
  vineyard-pipeline:dev \
  bash /app/azure_platform/run_all.sh
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `RUN_ID` | Yes (for job2/3) | auto-generated | Unique run identifier |
| `INPUT_MS` | Yes (for job1) | - | Path to input LAS/LAZ file |
| `OUT_BASE` | No | `/data` | Base path for silver/gold outputs |
| `VINEYARD_ID` | No | `default_vineyard` | Vineyard identifier |
| `FLIGHT_ID` | No | `flight_001` | Flight identifier |

## Jobs

### Job 1: Ground Removal (`job1_ground_removal.sh`)

Runs SMRF ground classification using existing `scripts/smrf_ground_classification.py`.

**Input:**
- LAS/LAZ file (multispectral point cloud)

**Output:**
- `silver/run_id=<run_id>/ms/non_ground.laz`
- `silver/run_id=<run_id>/ms/ground.laz`
- `silver/run_id=<run_id>/stage=ground/_SUCCESS.json`

### Job 2: Clustering (`job2_clustering.sh`)

Runs Euclidean cluster extraction using existing `scripts/build/clustering_only` (C++).

**Input:**
- `silver/run_id=<run_id>/ms/non_ground.laz`

**Output:**
- `silver/run_id=<run_id>/rows/row_index.parquet`
- `silver/run_id=<run_id>/clusters/cluster_*.laz`
- `silver/run_id=<run_id>/stage=cluster/_SUCCESS.json`

### Job 3: Features (`job3_features.sh`)

Computes NDVI and other features for each vineyard row.

**Input:**
- `silver/run_id=<run_id>/rows/row_index.parquet`
- `silver/run_id=<run_id>/clusters/*.laz`

**Output:**
- `gold/run_id=<run_id>/features/features_per_row.parquet`
- `gold/run_id=<run_id>/features/qc_summary.json`
- `gold/run_id=<run_id>/stage=features/_SUCCESS.json`

## Folder Structure

```
bronze/                                    # Raw immutable data
  vineyard_id=.../
    flight_date=YYYY-MM-DD/
      flight_id=.../
        sensors/ms/input.las

silver/                                    # Processed data (run-scoped)
  run_id=<run_id>/
    ms/
      non_ground.laz
      ground.laz
    rows/
      row_index.parquet
    clusters/
      cluster_00.laz
      cluster_01.laz
      ...
    stage=ground/_SUCCESS.json
    stage=cluster/_SUCCESS.json

gold/                                      # Final products (run-scoped)
  run_id=<run_id>/
    features/
      features_per_row.parquet
      qc_summary.json
    stage=features/_SUCCESS.json
```

## Data Contracts

See `contracts.md` for detailed schema definitions:
- `row_index.parquet` - Output of Job 2, input to Job 3
- `features_per_row.parquet` - Final output with per-row features
- `_SUCCESS.json` - Success markers for orchestration

## Allowed Data Sources

Only files under `/datasource/flights/`:
- `07-15-MS.laz` - Multispectral (July 15)
- `08-19-MS.laz` - Multispectral (August 19)
- `07-15-LIDAR.laz` - LiDAR (July 15)
- `2025-07-15-IR.laz` - Infrared (July 15)

## Azure Migration Notes

For Azure Data Factory (ADF) orchestration:
1. Each job checks for the previous stage's `_SUCCESS.json` before running
2. Jobs write outputs to run-scoped paths (never overwrite other runs)
3. Jobs never delete global folders
4. Success markers enable simple "if exists" orchestration logic

## Dependencies

- Python 3.10+ with: `laspy`, `pandas`, `pyarrow`, `numpy`
- PDAL (CLI or Python bindings)
- C++ build tools (for clustering_only)
