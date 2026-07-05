#!/usr/bin/env bash
# make_snapshots.sh — render one PNG per pipeline checkpoint into
# scripts/pipeline_snapshots/. Read-only: it consumes the outputs that
# run_pipeline.sh already produced (out_ground/, out_cluster_las/) and the raw
# input cloud. It does NOT run or change the pipeline.
#
# Usage:
#   ./make_snapshots.sh                 # use pipeline defaults
#   ./make_snapshots.sh /path/raw.las   # override the raw input cloud
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA=()
if [ "$#" -ge 1 ]; then
  EXTRA+=(--raw-las "$1")
fi

python3 "$ROOT_DIR/visualization/pipeline_snapshots.py" "${EXTRA[@]}"
