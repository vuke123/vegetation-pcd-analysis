"""Runtime configuration — paths and knobs."""
from __future__ import annotations

import os
from pathlib import Path

# vineyard_app/backend/app/config.py -> vineyard_app/
APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT.parent

PIPELINE_SCRIPTS_DIR = Path(
    os.environ.get(
        "VINEYARD_PIPELINE_DIR",
        REPO_ROOT / "vegetation-pcd-analysis" / "scripts",
    )
).resolve()

PIPELINE_SCRIPT = PIPELINE_SCRIPTS_DIR / "run_pipeline.sh"

DATA_DIR = Path(os.environ.get("VINEYARD_DATA_DIR", APP_ROOT / "data")).resolve()
JOBS_DIR = DATA_DIR / "jobs"

MAX_UPLOAD_BYTES = int(os.environ.get("VINEYARD_MAX_UPLOAD_BYTES", 4 * 1024 * 1024 * 1024))
MAX_POINTS_PER_CLUSTER = int(os.environ.get("VINEYARD_MAX_POINTS_PER_CLUSTER", 8000))

JOBS_DIR.mkdir(parents=True, exist_ok=True)
