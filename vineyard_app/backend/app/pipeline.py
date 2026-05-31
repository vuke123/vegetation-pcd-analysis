"""Job manager: runs the vineyard pipeline as a subprocess, one at a time.

The upstream pipeline (`run_pipeline.sh`) writes outputs into its own
scripts directory (`out_ground/`, `out_cluster/`, `out_cluster_las/`).
After each run we copy the relevant artifacts into the app's per-job
data directory so results are preserved across jobs.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .config import JOBS_DIR, PIPELINE_SCRIPT, PIPELINE_SCRIPTS_DIR

JobStatus = str  # "queued" | "running" | "succeeded" | "failed"


@dataclass
class Job:
    id: str
    input_las: Path
    job_dir: Path
    status: JobStatus = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    log: str = ""
    error: Optional[str] = None

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "log": self.log,
            "error": self.error,
            "has_results": (self.job_dir / "clusters" / "row_features.parquet").exists(),
        }


class JobManager:
    """Serializes pipeline runs so they don't clobber each other's output dirs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._run_lock = threading.Lock()

    def create_job(self, input_las_source: Path, original_name: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOBS_DIR / job_id
        (job_dir / "clusters").mkdir(parents=True, exist_ok=True)
        # Move (not copy) — the upload was streamed to a temp path we own.
        # The upstream C++ `clustering_only` binary hardcodes the non-ground filename
        # stem `2025-07-15-MS_Vinograd_1`, so SMRF must emit that stem.
        input_path = job_dir / "2025-07-15-MS_Vinograd_1.las"
        shutil.move(str(input_las_source), str(input_path))
        (job_dir / "source_name.txt").write_text(original_name)

        job = Job(id=job_id, input_las=input_path, job_dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [j.to_public() for j in sorted(self._jobs.values(), key=lambda j: -j.created_at)]

    def _run(self, job: Job) -> None:
        with self._run_lock:
            job.status = "running"
            job.started_at = time.time()
            try:
                self._execute(job)
                job.status = "succeeded"
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.error = str(exc)
                job.log += f"\n\n[ERROR] {exc}\n"
            finally:
                job.finished_at = time.time()

    def _execute(self, job: Job) -> None:
        if not PIPELINE_SCRIPT.exists():
            raise RuntimeError(f"pipeline script not found: {PIPELINE_SCRIPT}")

        cmd = ["bash", str(PIPELINE_SCRIPT), str(job.input_las)]
        job.log += f"$ {' '.join(cmd)}\n(cwd: {PIPELINE_SCRIPTS_DIR})\n\n"

        proc = subprocess.Popen(
            cmd,
            cwd=str(PIPELINE_SCRIPTS_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            job.log += line
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"pipeline exited with status {proc.returncode}")

        self._collect_outputs(job)

    def _collect_outputs(self, job: Job) -> None:
        src = PIPELINE_SCRIPTS_DIR / "out_cluster_las"
        dst = job.job_dir / "clusters"
        dst.mkdir(parents=True, exist_ok=True)

        parquet = src / "row_features.parquet"
        if not parquet.exists():
            raise RuntimeError(f"expected output missing: {parquet}")
        shutil.copy2(parquet, dst / "row_features.parquet")

        for las in sorted(src.glob("*_cluster_*_ndvi.las")):
            shutil.copy2(las, dst / las.name)

        merged = src / "merged.las"
        if merged.exists():
            shutil.copy2(merged, dst / "merged.las")

        job.log += f"\n[ok] collected results into {dst}\n"


manager = JobManager()
