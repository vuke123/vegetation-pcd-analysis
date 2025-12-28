#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

INPUT_LAS_DEFAULT="../datasource/2025-07-15-MS_Vinograd_1.las"
INPUT_LAS="${1:-$INPUT_LAS_DEFAULT}"

OUT_GROUND_DIR="./out_ground"
OUT_CLUSTER_DIR="./out_cluster"
OUT_CLUSTER_LAS_DIR="./out_cluster_las"
FINAL_LAS="$OUT_CLUSTER_LAS_DIR/merged_clusters_ndvi.las"

echo "=== [0/5] Cleaning output folders (out_ground, out_cluster, out_cluster_las) ==="
rm -rf -- "$OUT_GROUND_DIR"/* "$OUT_GROUND_DIR"/.[!.]* "$OUT_GROUND_DIR"/..?* 2>/dev/null || true
rm -rf -- "$OUT_CLUSTER_DIR"/* "$OUT_CLUSTER_DIR"/.[!.]* "$OUT_CLUSTER_DIR"/..?* 2>/dev/null || true
rm -rf -- "$OUT_CLUSTER_LAS_DIR"/* "$OUT_CLUSTER_LAS_DIR"/.[!.]* "$OUT_CLUSTER_LAS_DIR"/..?* 2>/dev/null || true

echo "=== [1/5] Running SMRF ground classification (Python) ==="
NON_GROUND_LAS=$(
  INPUT_LAS="$INPUT_LAS" OUT_GROUND_DIR="$OUT_GROUND_DIR" \
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

sys.stdout = sys.__stdout__
print(non_ground, end="")
PY
)
echo "Non-ground LAS: ${NON_GROUND_LAS}"

echo "=== [2/5] Building C++ targets (cmake --build build -j) ==="
cmake --build build -j

echo "=== [3/5] Running clustering_only (C++) ==="
NON_GROUND_LAS="$NON_GROUND_LAS" ./build/clustering_only

echo "=== [4/5] Computing NDVI LAS for each cluster PCD ==="
mkdir -p "$OUT_CLUSTER_LAS_DIR"
shopt -s nullglob
template_las="$NON_GROUND_LAS"
cluster_pcds=("$OUT_CLUSTER_DIR"/config*_cluster_*.pcd)
if [ "${#cluster_pcds[@]}" -eq 0 ]; then
  echo "No cluster PCD files found in ${OUT_CLUSTER_DIR}"
  exit 1
fi

for pcd in "${cluster_pcds[@]}"; do
  base="$(basename "$pcd" .pcd)"
  out_las="${OUT_CLUSTER_LAS_DIR}/${base}_ndvi.las"
  echo "  - ${pcd} -> ${out_las}"
  python3 pcd_to_ndvi_las.py \
    --pcd "$pcd" \
    --template-las "$template_las" \
    --out-las "$out_las"
done

echo "=== [5/5] Merge clusters into one .LAS file ==="

set -euo pipefail

(
  cd "./out_cluster_las"
  pdal merge \
    --writers.las.extra_dims=all \
    --writers.las.minor_version=4 \
    *.las merged.las
)