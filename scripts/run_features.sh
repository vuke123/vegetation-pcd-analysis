#!/usr/bin/env bash
set -euo pipefail

# Feature-computation pipeline — runs ONLY on already-generated cluster data.
# Use this after a full run_pipeline.sh has produced out_cluster_las/*.las files.
# Add new feature steps here as they are implemented.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INPUT_LAS_DEFAULT="../datasource/2025-07-15-MS_Vinograd_1.las"
INPUT_LAS="${1:-$INPUT_LAS_DEFAULT}"

OUT_CLUSTER_LAS_DIR="./out_cluster_las"

if [ ! -d "$OUT_CLUSTER_LAS_DIR" ] || [ -z "$(ls -A "$OUT_CLUSTER_LAS_DIR"/*.las 2>/dev/null)" ]; then
  echo "ERROR: No LAS files found in $OUT_CLUSTER_LAS_DIR. Run run_pipeline.sh first." >&2
  exit 1
fi

echo "=== [6/7] Compute per-row features ==="
python3 compute_row_features.py \
  --in-dir "$OUT_CLUSTER_LAS_DIR" \
  --source-las "$INPUT_LAS" \
  --out "$OUT_CLUSTER_LAS_DIR/row_features.parquet"

echo "=== [7/7] Compute canopy structure metrics (segment-based) ==="
python3 compute_canopy_structure.py \
  --in-dir "$OUT_CLUSTER_LAS_DIR" \
  --source-las "$INPUT_LAS" \
  --segment-length 1.0 \
  --out "$OUT_CLUSTER_LAS_DIR/row_canopy_structure.parquet"

echo "=== Done. Outputs: ==="
echo "  $OUT_CLUSTER_LAS_DIR/row_features.parquet"
echo "  $OUT_CLUSTER_LAS_DIR/row_canopy_structure.parquet"
