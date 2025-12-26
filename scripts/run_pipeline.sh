#!/usr/bin/env bash
set -euo pipefail

# Root of the DIPLOMSKI-RAD project (this script location)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "=== [0/4] Cleaning output folders (out_cluster, out_ground) ==="
rm -rf -- ./out_cluster/* ./out_cluster/.[!.]* ./out_cluster/..?* \
          ./out_ground/*  ./out_ground/.[!.]*  ./out_ground/..?* 2>/dev/null || true
          
echo "=== [1/4] Building C++ targets (cmake --build build -j) ==="
cmake --build build -j

echo "=== [2/4] Running ground_removal_only ==="
./build/ground_removal_only

echo "=== [3/4] Running clustering_only ==="
./build/clustering_only

echo "=== [4/4] Launching Python visualization ==="
python3 ./visualize_tutorial_clusters.py
