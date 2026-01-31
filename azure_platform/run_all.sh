#!/usr/bin/env bash
# =============================================================================
# Run All Jobs (Local/Container Pipeline)
# =============================================================================
# Runs all three jobs in sequence for local testing.
# For Azure, each job would be triggered independently by ADF.
#
# Usage (local):
#   ./run_all.sh --input <path> [--run-id <run_id>] [--base-path <path>]
#
# Usage (Docker):
#   docker run --rm -v "$(pwd)":/data \
#     -e RUN_ID="test_001" \
#     -e INPUT_MS="/data/datasource/flights/07-15-MS.laz" \
#     -e OUT_BASE="/data" \
#     vineyard-pipeline:dev bash /app/azure_platform/run_all.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Detect if running in container or locally
if [[ -d "/app/scripts" ]]; then
  DEFAULT_BASE="/data"
else
  DEFAULT_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
INPUT_MS="${INPUT_MS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$(head -c 4 /dev/urandom | xxd -p)}"
OUT_BASE="${OUT_BASE:-$DEFAULT_BASE}"
BASE_PATH="${OUT_BASE}"

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
  echo "Usage: $0 --input <path>"
  exit 1
fi

echo "=============================================="
echo "Running Full Pipeline"
echo "=============================================="
echo "Run ID:     $RUN_ID"
echo "Input:      $INPUT_MS"
echo "Base path:  $BASE_PATH"
echo "=============================================="

# Job 1: Ground Removal
"$SCRIPT_DIR/job1_ground_removal.sh" \
  --input "$INPUT_MS" \
  --run-id "$RUN_ID" \
  --base-path "$BASE_PATH"

# Job 2: Clustering
"$SCRIPT_DIR/job2_clustering.sh" \
  --run-id "$RUN_ID" \
  --base-path "$BASE_PATH"

# Job 3: Features
"$SCRIPT_DIR/job3_features.sh" \
  --run-id "$RUN_ID" \
  --base-path "$BASE_PATH"

echo "=============================================="
echo "Pipeline Complete"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo ""
echo "Outputs:"
echo "  Silver: $BASE_PATH/silver/run_id=$RUN_ID/"
echo "  Gold:   $BASE_PATH/gold/run_id=$RUN_ID/"
