#!/usr/bin/env bash
# =============================================================================
# Job 1: Ground Removal
# =============================================================================
# Runs SMRF ground classification and splits into ground/non-ground.
#
# Input:  INPUT_MS (LAS/LAZ file path)
# Output: silver/run_id=<run_id>/ms/non_ground.laz
#         silver/run_id=<run_id>/stage=ground/_SUCCESS.json
#
# Usage (local):
#   ./job1_ground_removal.sh --input <path> --run-id <run_id> [--base-path <path>]
#
# Usage (Docker):
#   docker run --rm -v "$(pwd)":/data \
#     -e RUN_ID="test_001" \
#     -e INPUT_MS="/data/datasource/flights/07-15-MS.laz" \
#     -e OUT_BASE="/data" \
#     vineyard-pipeline:dev bash /app/azure_platform/job1_ground_removal.sh
#
# Environment variables:
#   INPUT_MS      - Input LAS/LAZ file path (required)
#   RUN_ID        - Run identifier (auto-generated if not set)
#   OUT_BASE      - Base path for silver/gold outputs (default: project root or /data)
#   VINEYARD_ID   - Vineyard identifier (optional, for metadata)
#   FLIGHT_ID     - Flight identifier (optional, for metadata)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if running in container or locally
if [[ -d "/app/scripts" ]]; then
  # Container mode
  SCRIPTS_DIR="/app/scripts"
  DEFAULT_BASE="/data"
else
  # Local mode
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  SCRIPTS_DIR="$PROJECT_ROOT/scripts"
  DEFAULT_BASE="$PROJECT_ROOT"
fi

# -----------------------------------------------------------------------------
# Parse arguments (env vars take precedence, then CLI args)
# -----------------------------------------------------------------------------
INPUT_MS="${INPUT_MS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$(head -c 4 /dev/urandom | xxd -p)}"
OUT_BASE="${OUT_BASE:-$DEFAULT_BASE}"
BASE_PATH="${OUT_BASE}"  # Alias for compatibility

while [[ $# -gt 0 ]]; do
  case $1 in
    --input|-i)
      INPUT_MS="${INPUT_MS:-$2}"
      shift 2
      ;;
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

if [[ -z "$INPUT_MS" ]]; then
  echo "Error: INPUT_MS is required (via env var or --input)"
  echo "Usage: $0 --input <path> --run-id <run_id>"
  exit 1
fi

if [[ ! -f "$INPUT_MS" ]]; then
  echo "Error: Input file not found: $INPUT_MS"
  exit 1
fi

# -----------------------------------------------------------------------------
# Define output paths (silver layer)
# -----------------------------------------------------------------------------
SILVER_DIR="$BASE_PATH/silver/run_id=$RUN_ID"
OUT_DIR="$SILVER_DIR/ms"
SUCCESS_DIR="$SILVER_DIR/stage=ground"

mkdir -p "$OUT_DIR"
mkdir -p "$SUCCESS_DIR"

echo "=== Job 1: Ground Removal ==="
echo "Run ID:     $RUN_ID"
echo "Input:      $INPUT_MS"
echo "Output dir: $OUT_DIR"

START_TIME=$(date +%s)

# -----------------------------------------------------------------------------
# Run SMRF ground classification using existing Python script
# -----------------------------------------------------------------------------
cd "$SCRIPTS_DIR"

NON_GROUND_LAS=$(
  INPUT_LAS="$INPUT_MS" OUT_GROUND_DIR="$OUT_DIR" \
  python3 - << 'PY'
import os
import sys
from smrf_ground_classification import run_smrf_classification, split_ground_non_ground

class StdoutToStderr:
    def write(self, s: str) -> None:
        sys.stderr.write(s)
    def flush(self) -> None:
        sys.stderr.flush()

sys.stdout = StdoutToStderr()

input_las = os.environ["INPUT_LAS"]
out_dir = os.environ["OUT_GROUND_DIR"]

classified = run_smrf_classification(input_las, out_dir=out_dir)
ground, non_ground = split_ground_non_ground(classified, out_dir)

# Clean up intermediate classified file
import os as os_mod
if os_mod.path.exists(classified):
    os_mod.remove(classified)

sys.stdout = sys.__stdout__
print(non_ground, end="")
PY
)

echo "Non-ground output: $NON_GROUND_LAS"

# -----------------------------------------------------------------------------
# Rename to standard convention: non_ground.laz
# -----------------------------------------------------------------------------
FINAL_OUTPUT="$OUT_DIR/non_ground.laz"

if [[ "$NON_GROUND_LAS" != "$FINAL_OUTPUT" ]]; then
  # Convert to LAZ if needed, or just rename
  if [[ "$NON_GROUND_LAS" == *.las ]]; then
    echo "Converting to LAZ..."
    pdal translate "$NON_GROUND_LAS" "$FINAL_OUTPUT" \
      --writers.las.compression=laszip \
      --writers.las.minor_version=4 \
      --writers.las.forward=all \
      --writers.las.extra_dims=all
    rm -f "$NON_GROUND_LAS"
  else
    mv "$NON_GROUND_LAS" "$FINAL_OUTPUT"
  fi
fi

# Also handle ground file
GROUND_FILES=("$OUT_DIR"/*_ground.las "$OUT_DIR"/*_ground.laz)
for gf in "${GROUND_FILES[@]}"; do
  if [[ -f "$gf" ]]; then
    GROUND_OUTPUT="$OUT_DIR/ground.laz"
    if [[ "$gf" == *.las ]]; then
      pdal translate "$gf" "$GROUND_OUTPUT" \
        --writers.las.compression=laszip \
        --writers.las.minor_version=4 \
        --writers.las.forward=all \
        --writers.las.extra_dims=all
      rm -f "$gf"
    else
      mv "$gf" "$GROUND_OUTPUT"
    fi
    break
  fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# -----------------------------------------------------------------------------
# Write success marker
# -----------------------------------------------------------------------------
cat > "$SUCCESS_DIR/_SUCCESS.json" << EOF
{
  "stage": "ground",
  "run_id": "$RUN_ID",
  "timestamp": "$(date -Iseconds)",
  "input_files": ["$INPUT_MS"],
  "output_files": ["$FINAL_OUTPUT"],
  "metrics": {
    "duration_s": $DURATION
  },
  "errors": []
}
EOF

echo "=== Job 1: Ground Removal completed ==="
echo "Output:  $FINAL_OUTPUT"
echo "Success: $SUCCESS_DIR/_SUCCESS.json"
echo "Duration: ${DURATION}s"
