#!/usr/bin/env bash
# =============================================================================
# Job 2: Clustering (PCD -> LAZ via Python)
# =============================================================================
# Runs Euclidean clustering (clustering_only) and then packages clusters into
# per-row LAZ files using pcd_to_ndvi_las.py (laspy), avoiding PDAL cast issues.
#
# Input:
#   <OUT_BASE>/silver/run_id=<run_id>/ms/non_ground.laz
#   <OUT_BASE>/silver/run_id=<run_id>/stage=ground/_SUCCESS.json
#
# Output:
#   <OUT_BASE>/silver/run_id=<run_id>/clusters/config*_cluster_*.pcd        (intermediate)
#   <OUT_BASE>/silver/run_id=<run_id>/clusters/cluster_00.laz ...           (final cluster files)
#   <OUT_BASE>/silver/run_id=<run_id>/rows/row_index.parquet               (contract)
#   <OUT_BASE>/silver/run_id=<run_id>/stage=cluster/_SUCCESS.json
#
# Docker:
#   docker run --rm -v "$(pwd)":/data \
#     -e RUN_ID="test03" \
#     -e OUT_BASE="/data/out" \
#     vineyard-pipeline:dev bash /app/azure_platform/job2_clustering.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if running in container or locally
if [[ -d "/app/scripts" ]]; then
  DEFAULT_BASE="/data"
  SCRIPTS_DIR="/app/scripts"
  CLUSTERING_BIN="/app/bin/clustering_only"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  DEFAULT_BASE="$PROJECT_ROOT"
  SCRIPTS_DIR="$PROJECT_ROOT/scripts"
  CLUSTERING_BIN="$PROJECT_ROOT/build/clustering_only"
fi

# -----------------------------------------------------------------------------
# Parse args/env
# -----------------------------------------------------------------------------
RUN_ID="${RUN_ID:-}"
OUT_BASE="${OUT_BASE:-$DEFAULT_BASE}"
BASE_PATH="${OUT_BASE}"

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "Error: RUN_ID is required"
  exit 1
fi

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SILVER_DIR="$BASE_PATH/silver/run_id=$RUN_ID"
GROUND_SUCCESS="$SILVER_DIR/stage=ground/_SUCCESS.json"
INPUT_LAZ="$SILVER_DIR/ms/non_ground.laz"

CLUSTERS_DIR="$SILVER_DIR/clusters"
ROWS_DIR="$SILVER_DIR/rows"
SUCCESS_DIR="$SILVER_DIR/stage=cluster"

mkdir -p "$CLUSTERS_DIR" "$ROWS_DIR" "$SUCCESS_DIR"

echo "=== Job 2: Clustering (PCD -> LAZ via Python) ==="
echo "Run ID:      $RUN_ID"
echo "Base path:   $BASE_PATH"
echo "Input:       $INPUT_LAZ"
echo "Clusters:    $CLUSTERS_DIR"
echo "Row index:   $ROWS_DIR/row_index.parquet"
echo "Binary:      $CLUSTERING_BIN"
echo "Scripts:     $SCRIPTS_DIR"

# -----------------------------------------------------------------------------
# Prerequisites
# -----------------------------------------------------------------------------
if [[ ! -f "$GROUND_SUCCESS" ]]; then
  echo "Error: Job 1 not completed for this RUN_ID."
  echo "Missing: $GROUND_SUCCESS"
  exit 1
fi

if [[ ! -f "$INPUT_LAZ" ]]; then
  echo "Error: Input file not found: $INPUT_LAZ"
  exit 1
fi

if [[ ! -f "$CLUSTERING_BIN" ]]; then
  echo "Error: clustering_only binary not found: $CLUSTERING_BIN"
  exit 1
fi

if [[ ! -f "$SCRIPTS_DIR/pcd_to_ndvi_las.py" ]]; then
  echo "Error: Missing converter script: $SCRIPTS_DIR/pcd_to_ndvi_las.py"
  exit 1
fi

START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Run clustering_only (writes PCD clusters)
# -----------------------------------------------------------------------------
cd "$SCRIPTS_DIR"

# clustering_only reads these env vars (per your edited C++)
export NON_GROUND_LAS="$INPUT_LAZ"
export OUT_CLUSTER_DIR="$CLUSTERS_DIR"

echo "Running clustering_only with:"
echo "  NON_GROUND_LAS=$NON_GROUND_LAS"
echo "  OUT_CLUSTER_DIR=$OUT_CLUSTER_DIR"

"$CLUSTERING_BIN"

# -----------------------------------------------------------------------------
# Convert PCD clusters -> LAZ using Python converter
# -----------------------------------------------------------------------------
echo "Packaging clusters: PCD -> LAZ using pcd_to_ndvi_las.py ..."
shopt -s nullglob
cluster_pcds=("$CLUSTERS_DIR"/config*_cluster_*.pcd)

if [[ ${#cluster_pcds[@]} -eq 0 ]]; then
  echo "Error: No cluster PCD files found in $CLUSTERS_DIR"
  echo "Expected: config*_cluster_*.pcd"
  exit 1
fi

# Remove old cluster_*.laz from previous attempts to avoid mixing
rm -f "$CLUSTERS_DIR"/cluster_*.laz 2>/dev/null || true

CLUSTER_COUNT=0
for pcd in "${cluster_pcds[@]}"; do
  out_laz="$CLUSTERS_DIR/cluster_$(printf '%02d' $CLUSTER_COUNT).laz"

  python3 "$SCRIPTS_DIR/pcd_to_ndvi_las.py" \
    --pcd "$pcd" \
    --template-las "$INPUT_LAZ" \
    --out-las "$out_laz"

  echo "  $pcd -> $out_laz"
  CLUSTER_COUNT=$((CLUSTER_COUNT + 1))
done

# -----------------------------------------------------------------------------
# Build row_index.parquet from cluster_*.laz
# -----------------------------------------------------------------------------
echo "Building row_index.parquet..."

python3 - << PYEOF
from pathlib import Path
import pandas as pd
import laspy

run_id = "$RUN_ID"
clusters_dir = Path("$CLUSTERS_DIR")
output_path = Path("$ROWS_DIR/row_index.parquet")

cluster_files = sorted(clusters_dir.glob("cluster_*.laz"))
if not cluster_files:
    raise SystemExit(f"No cluster_*.laz files found in {clusters_dir}")

rows = []
for cluster_path in cluster_files:
    row_id = int(cluster_path.stem.split("_")[1])
    with laspy.open(cluster_path) as f:
        las = f.read()

    rows.append({
        "run_id": run_id,
        "row_id": row_id,
        "point_count": len(las.x),
        "bbox_minx": float(las.x.min()),
        "bbox_miny": float(las.y.min()),
        "bbox_minz": float(las.z.min()),
        "bbox_maxx": float(las.x.max()),
        "bbox_maxy": float(las.y.max()),
        "bbox_maxz": float(las.z.max()),
        "crs_epsg": None,
        "crs_wkt": None,
        "centerline_wkt": None,
        "poly_wkt": None,
        "cluster_file": str(cluster_path),
    })

df = pd.DataFrame(rows)
df.to_parquet(output_path, index=False)
print(f"Wrote {len(df)} rows to {output_path}")
PYEOF

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# -----------------------------------------------------------------------------
# Success marker
# -----------------------------------------------------------------------------
cat > "$SUCCESS_DIR/_SUCCESS.json" << EOF
{
  "stage": "cluster",
  "run_id": "$RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "input_files": ["$INPUT_LAZ"],
  "output_files": ["$ROWS_DIR/row_index.parquet", "$CLUSTERS_DIR/cluster_*.laz"],
  "metrics": {
    "duration_s": $DURATION,
    "num_clusters": $CLUSTER_COUNT
  },
  "errors": []
}
EOF

echo "=== Job 2 completed ==="
echo "Clusters:    $CLUSTER_COUNT"
echo "Row index:   $ROWS_DIR/row_index.parquet"
echo "Success:     $SUCCESS_DIR/_SUCCESS.json"
echo "Duration:    ${DURATION}s"
