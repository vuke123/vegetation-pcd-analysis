#!/usr/bin/env bash
# =============================================================================
# Job 3: Feature Extraction
# =============================================================================
# Computes NDVI and optional volume/IR features for each vineyard row.
#
# Input:  <OUT_BASE>/silver/run_id=<run_id>/rows/row_index.parquet
#         <OUT_BASE>/silver/run_id=<run_id>/clusters/*.laz  (via row_index.cluster_file)
#
# Output: <OUT_BASE>/gold/run_id=<run_id>/features/features_per_row.parquet
#         <OUT_BASE>/gold/run_id=<run_id>/features/qc_summary.json
#         <OUT_BASE>/gold/run_id=<run_id>/stage=features/_SUCCESS.json
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if running in container or locally
if [[ -d "/app/scripts" ]]; then
  SCRIPTS_DIR="/app/scripts"
  DEFAULT_BASE="/data"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  SCRIPTS_DIR="$PROJECT_ROOT/scripts"
  DEFAULT_BASE="$PROJECT_ROOT"
fi

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
RUN_ID="${RUN_ID:-}"
OUT_BASE="${OUT_BASE:-$DEFAULT_BASE}"
BASE_PATH="${OUT_BASE}"
VINEYARD_ID="${VINEYARD_ID:-default_vineyard}"
FLIGHT_ID="${FLIGHT_ID:-flight_001}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-id|-r)
      RUN_ID="${RUN_ID:-$2}"
      shift 2
      ;;
    --base-path|-b)
      BASE_PATH="$2"
      OUT_BASE="$2"
      shift 2
      ;;
    --vineyard-id)
      VINEYARD_ID="$2"
      shift 2
      ;;
    --flight-id)
      FLIGHT_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "Error: RUN_ID is required (via env var or --run-id)"
  exit 1
fi

# -----------------------------------------------------------------------------
# Check prerequisites (must match Job2 output layout)
# -----------------------------------------------------------------------------
SILVER_DIR="$BASE_PATH/silver/run_id=$RUN_ID"
CLUSTER_SUCCESS="$SILVER_DIR/stage=cluster/_SUCCESS.json"

if [[ ! -f "$CLUSTER_SUCCESS" ]]; then
  echo "Error: Job 2 (clustering) has not completed."
  echo "Missing: $CLUSTER_SUCCESS"
  exit 1
fi

ROW_INDEX="$SILVER_DIR/rows/row_index.parquet"
if [[ ! -f "$ROW_INDEX" ]]; then
  echo "Error: Row index not found: $ROW_INDEX"
  exit 1
fi

CLUSTERS_DIR="$SILVER_DIR/clusters"
if [[ ! -d "$CLUSTERS_DIR" ]]; then
  echo "Error: Clusters directory not found: $CLUSTERS_DIR"
  exit 1
fi

# -----------------------------------------------------------------------------
# Define output paths (gold layer)
# -----------------------------------------------------------------------------
GOLD_DIR="$BASE_PATH/gold/run_id=$RUN_ID"
FEATURES_DIR="$GOLD_DIR/features"
SUCCESS_DIR="$GOLD_DIR/stage=features"

mkdir -p "$FEATURES_DIR" "$SUCCESS_DIR"

echo "=== Job 3: Feature Extraction ==="
echo "Run ID:      $RUN_ID"
echo "Base path:   $BASE_PATH"
echo "Vineyard:    $VINEYARD_ID"
echo "Flight:      $FLIGHT_ID"
echo "Row index:   $ROW_INDEX"
echo "Clusters:    $CLUSTERS_DIR"
echo "Output:      $FEATURES_DIR/features_per_row.parquet"

START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Compute NDVI features per row (robust cluster_file resolution)
# -----------------------------------------------------------------------------
cd "$SCRIPTS_DIR"

python3 - << PYEOF
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import laspy

run_id = "$RUN_ID"
vineyard_id = "$VINEYARD_ID"
flight_id = "$FLIGHT_ID"
row_index_path = Path("$ROW_INDEX")
clusters_dir = Path("$CLUSTERS_DIR")
features_output = Path("$FEATURES_DIR/features_per_row.parquet")
qc_output = Path("$FEATURES_DIR/qc_summary.json")

row_index = pd.read_parquet(row_index_path)
print(f"Processing {len(row_index)} rows...")

features_rows = []
all_warnings = []

def resolve_cluster_path(p: str) -> Path:
    cp = Path(p)
    if cp.is_file():
        return cp
    # if relative or broken absolute, try resolving under clusters_dir
    cp2 = clusters_dir / cp.name
    return cp2

for _, r in row_index.iterrows():
    row_id = int(r["row_id"])
    row_warnings = []

    raw_path = str(r.get("cluster_file", ""))
    if not raw_path:
        # fallback: look for cluster_<id>.laz
        candidate = clusters_dir / f"cluster_{row_id:02d}.laz"
        cluster_path = candidate
        row_warnings.append(f"Row {row_id}: cluster_file missing in row_index; using {candidate.name}")
    else:
        cluster_path = resolve_cluster_path(raw_path)

    if not cluster_path.is_file():
        msg = f"Row {row_id}: Cluster file not found: {cluster_path}"
        row_warnings.append(msg)
        all_warnings.append(msg)
        continue

    with laspy.open(cluster_path) as f:
        las = f.read()

    # laspy: common names are 'red' and 'infrared' (sometimes 'nir' depending on source)
    has_red = hasattr(las, "red")
    has_infrared = hasattr(las, "infrared")
    has_nir = hasattr(las, "nir")

    if has_red and (has_infrared or has_nir):
        red = np.array(las.red, dtype=np.float64)
        nir = np.array(las.nir if has_nir else las.infrared, dtype=np.float64)

        eps = 1e-6
        ndvi = (nir - red) / (nir + red + eps)

        ndvi_mean = float(np.mean(ndvi))
        ndvi_std = float(np.std(ndvi))
        ndvi_p10 = float(np.percentile(ndvi, 10))
        ndvi_p90 = float(np.percentile(ndvi, 90))
        ndvi_low_frac = float(np.mean(ndvi < 0.2))
    else:
        ndvi_mean = ndvi_std = ndvi_p10 = ndvi_p90 = ndvi_low_frac = 0.0
        row_warnings.append(f"Row {row_id}: Missing red+nir/infrared fields for NDVI")

    # aggregate warnings
    for w in row_warnings:
        all_warnings.append(w)

    features_rows.append({
        "run_id": run_id,
        "vineyard_id": vineyard_id,
        "flight_id": flight_id,
        "row_id": row_id,

        "ndvi_mean": ndvi_mean,
        "ndvi_std": ndvi_std,
        "ndvi_p10": ndvi_p10,
        "ndvi_p90": ndvi_p90,
        "ndvi_low_frac": ndvi_low_frac,

        "temp_mean": None,
        "temp_max": None,
        "temp_std": None,

        "vol_voxel": None,
        "vol_slicing": None,
        "vol_hull": None,
        "vol_disagreement": None,

        "seg_confidence": None,
        "gap_score": None,
        "warnings": json.dumps(row_warnings),
    })

    print(f"  Row {row_id}: NDVI mean={ndvi_mean:.3f}")

features_df = pd.DataFrame(features_rows)
features_df.to_parquet(features_output, index=False)
print(f"Wrote {len(features_df)} rows to {features_output}")

qc_summary = {
    "run_id": run_id,
    "total_rows_in_row_index": int(len(row_index)),
    "rows_written": int(len(features_df)),
    "rows_with_warnings": int((features_df["warnings"] != "[]").sum()) if len(features_df) else 0,
    "warnings": all_warnings,
}
with open(qc_output, "w") as f:
    json.dump(qc_summary, f, indent=2)
print(f"Wrote QC summary to {qc_output}")
PYEOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# -----------------------------------------------------------------------------
# Write success marker
# -----------------------------------------------------------------------------
cat > "$SUCCESS_DIR/_SUCCESS.json" << EOF
{
  "stage": "features",
  "run_id": "$RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "input_files": ["$ROW_INDEX"],
  "output_files": ["$FEATURES_DIR/features_per_row.parquet", "$FEATURES_DIR/qc_summary.json"],
  "metrics": {
    "duration_s": $DURATION
  },
  "errors": []
}
EOF

echo "=== Job 3: Feature Extraction completed ==="
echo "Features:    $FEATURES_DIR/features_per_row.parquet"
echo "QC Summary:  $FEATURES_DIR/qc_summary.json"
echo "Success:     $SUCCESS_DIR/_SUCCESS.json"
echo "Duration:    ${DURATION}s"
