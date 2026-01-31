"""
Azure Platform Pipeline for Vineyard Point Cloud Analysis.

This folder contains the refactored pipeline split into 3 independent jobs
for Azure migration. Each job can run independently with clean contracts.

Folder Structure (bronze/silver/gold data lake pattern):
    bronze/vineyard_id=.../flight_date=YYYY-MM-DD/flight_id=.../sensors/ms/input.las
    silver/run_id=<run_id>/ms/non_ground.laz
    silver/run_id=<run_id>/rows/row_index.parquet
    gold/run_id=<run_id>/features/features_per_row.parquet

Jobs:
    - job1_ground_removal.sh: SMRF ground classification → non_ground.laz
    - job2_clustering.sh: Euclidean clustering → row_index.parquet + per-row LAZ
    - job3_features.sh: NDVI/volume extraction → features_per_row.parquet

Success Markers:
    silver/run_id=.../stage=ground/_SUCCESS.json
    silver/run_id=.../stage=cluster/_SUCCESS.json
    gold/run_id=.../stage=features/_SUCCESS.json

Data Sources (allowed):
    /datasource/flights/07-15-MS.laz
    /datasource/flights/08-19-MS.laz
    /datasource/flights/07-15-LIDAR.laz
    /datasource/flights/2025-07-15-IR.laz
"""

__version__ = "0.1.0"
