#!/usr/bin/env bash
# =============================================================================
# Job 2 (Resume): Convert existing cluster PCDs to LAZ (via Python) + row_index
# =============================================================================
# Assumes clustering_only already produced:
#   <OUT_BASE>/silver/run_id=<run_id>/clusters/config*_cluster_*.pcd
#
# This resume script does NOT run clustering_only again.
# It converts PCD -> LAZ using pcd_to_ndvi_las.py (laspy), avoiding PDAL cast errors.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if running in container or locally
if [[ -d "/app/scripts" ]]; then
  DEFAULT_BASE="/data"
  SCRIPTS_DIR="/app/scripts"
else
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  DEFAULT_BASE="$PROJECT_ROOT"
  SCRIPTS_DIR="$PROJECT_ROOT/scripts"
fi

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

SILVER_DIR="$BASE_PATH/silver/run_id=$RUN_ID"
CLUSTERS_DIR="$SILVER_DIR/clusters"
ROWS_DIR="$SILVER_DIR/rows"
SUCCESS_DIR="$SILVER_DIR/stage=cluster"

mkdir -p "$CLUSTERS_DIR" "$ROWS_DIR" "$SUCCESS_DIR"

TEMPLATE_LAS="$SILVER_DIR/ms/non_ground.laz"
if [[ ! -f "$TEMPLATE_LAS" ]]; then
  echo "Error: Template LAS/LAZ not found: $TEMPLATE_LAS"
  echo "Run Job1 first (ground removal) for this RUN_ID."
  exit 1
fi

echo "=== Job2 Resume (Python): PCD -> LAZ + row_index ==="
echo "Run ID:      $RUN_ID"
echo "Base path:   $BASE_PATH"
echo "Clusters:    $CLUSTERS_DIR"
echo "Template:    $TEMPLATE_LAS"
echo "Row index:   $ROWS_DIR/row_index.parquet"

shopt -s nullglob
cluster_pcds=("$CLUSTERS_DIR"/config*_cluster_*.pcd)
if [[ ${#cluster_pcds[@]} -eq 0 ]]; then
  echo "Error: No cluster PCD files found in: $CLUSTERS_DIR"
  echo "Expected: config*_cluster_*.pcd"
  exit 1
fi

START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Convert PCD -> LAZ using your Python script (safe casting + optional NDVI extra dim)
# -----------------------------------------------------------------------------
echo "Converting ${#cluster_pcds[@]} PCD clusters to LAZ using pcd_to_ndvi_las.py..."

CLUSTER_COUNT=0
for pcd in "${cluster_pcds[@]}"; do
  out_laz="$CLUSTERS_DIR/cluster_$(printf '%02d' $CLUSTER_COUNT).laz"

  python3 "$SCRIPTS_DIR/pcd_to_ndvi_las.py" \
    --pcd "$pcd" \
    --template-las "$TEMPLATE_LAS" \
    --out-las "$out_laz"

  echo "  $pcd -> $out_laz"
  CLUSTER_COUNT=$((CLUSTER_COUNT + 1))
done

# -----------------------------------------------------------------------------
# Build row_index.parquet based on cluster_*.laz
# -----------------------------------------------------------------------------
echo "Building row_index.parquet..."

python3 - << PYEOF
from pathlib import Path
import pandas as pd
import laspy

run_id = "$RUN_ID"
clusters_dir = Path("$CLUSTERS_DIR")
output_path = Path("$ROWS_DIR/row_index.parquet")

rows = []
for cluster_path in sorted(clusters_dir.glob("cluster_*.laz")):
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
# Write success marker
# -----------------------------------------------------------------------------
cat > "$SUCCESS_DIR/_SUCCESS.json" << EOF
{
  "stage": "cluster",
  "run_id": "$RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "input_files": ["$CLUSTERS_DIR/config*_cluster_*.pcd"],
  "output_files": ["$ROWS_DIR/row_index.parquet", "$CLUSTERS_DIR/cluster_*.laz"],
  "metrics": {
    "duration_s": $DURATION,
    "num_clusters": $CLUSTER_COUNT
  },
  "errors": []
}
EOF

echo "=== Job2 Resume completed ==="
echo "Clusters:    $CLUSTER_COUNT"
echo "Row index:   $ROWS_DIR/row_index.parquet"
echo "Success:     $SUCCESS_DIR/_SUCCESS.json"
echo "Duration:    ${DURATION}s"
