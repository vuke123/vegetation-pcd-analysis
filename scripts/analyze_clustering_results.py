#!/usr/bin/env python3
"""
Analyze and visualize clustering results from different parameter configurations.
"""

import os
import glob
import re

def analyze_results():
    print("=" * 80)
    print("CLUSTERING RESULTS ANALYSIS")
    print("=" * 80)
    print()
    
    # Find all generated cluster files
    cluster_files = glob.glob("config*_cluster_*.pcd")
    
    if not cluster_files:
        print("No cluster files found! Run ./build/pcl_tutorial_clustering first.")
        return
    
    # Group by configuration
    configs = {}
    for filename in cluster_files:
        match = re.match(r'config(\d+)_cluster_(\d+)\.pcd', filename)
        if match:
            config_num = int(match.group(1))
            cluster_num = int(match.group(2))
            
            if config_num not in configs:
                configs[config_num] = []
            configs[config_num].append((cluster_num, filename))
    
    # Configuration parameters
    param_map = {
        1: {"leaf": 0.05, "dist": 0.1, "clust": 0.1},
        2: {"leaf": 0.05, "dist": 0.3, "clust": 0.3},
        3: {"leaf": 0.05, "dist": 0.8, "clust": 0.8},
        4: {"leaf": 0.20, "dist": 0.1, "clust": 0.1},
        5: {"leaf": 0.20, "dist": 0.3, "clust": 0.3},
        6: {"leaf": 0.20, "dist": 0.8, "clust": 0.8},
        7: {"leaf": 0.6, "dist": 0.1, "clust": 0.1},
        8: {"leaf": 0.6, "dist": 0.3, "clust": 0.3},
        9: {"leaf": 0.6, "dist": 0.8, "clust": 0.8},
    }
    
    # Print summary
    print(f"Found {len(configs)} configurations with clusters:\n")
    
    for config_num in sorted(configs.keys()):
        clusters = configs[config_num]
        params = param_map.get(config_num, {})
        
        print(f"Configuration #{config_num}:")
        print(f"  Parameters: leaf={params.get('leaf', '?')}m, "
              f"dist_thresh={params.get('dist', '?')}m, clust_tol={params.get('clust', '?')}m")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Files: {', '.join([f[1] for f in sorted(clusters)[:5]])}")
        if len(clusters) > 5:
            print(f"         ... and {len(clusters) - 5} more")
        print()
    
    print("=" * 80)
    print("RECOMMENDATIONS:")
    print("=" * 80)
    print()
    print("For vineyard analysis, look for configurations that produce:")
    print("  • 2-10 clusters (likely vine rows)")
    print("  • Large, coherent clusters (not fragmented)")
    print("  • Similar cluster sizes (uniform vine rows)")
    print()
    print("Avoid configurations with:")
    print("  • 100+ clusters (over-segmentation)")
    print("  • Very few (<2) or no clusters (under-segmentation)")
    print()
    
    # Find best candidates based on cluster count
    best_configs = []
    for config_num, clusters in configs.items():
        cluster_count = len(clusters)
        if 2 <= cluster_count <= 15:
            best_configs.append((config_num, cluster_count))
    
    if best_configs:
        print("LIKELY BEST CONFIGURATIONS (based on cluster count 2-15):")
        for config_num, count in sorted(best_configs, key=lambda x: abs(x[1] - 5)):
            params = param_map.get(config_num, {})
            print(f"  Config #{config_num}: {count} clusters "
                  f"[leaf={params.get('leaf', '?')}m, "
                  f"dist={params.get('dist', '?')}m, clust={params.get('clust', '?')}m]")
    else:
        print("⚠️  No configurations produced 2-15 clusters. Try different parameters.")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    analyze_results()
