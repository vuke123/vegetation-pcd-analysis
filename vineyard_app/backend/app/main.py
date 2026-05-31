"""FastAPI entrypoint for the vineyard web app."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import MAX_UPLOAD_BYTES
from .data import read_all_clusters, read_metrics
from .pipeline import manager
from .voxelize import voxelise_job

app = FastAPI(title="Vineyard PCD Analysis", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)) -> dict:
    name = file.filename or "upload.las"
    if not name.lower().endswith((".las", ".laz")):
        raise HTTPException(400, "file must be .las or .laz")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix)
    total = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
            tmp.write(chunk)
        tmp.close()
    except HTTPException:
        raise
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise

    job = manager.create_job(Path(tmp.name), original_name=name)
    return job.to_public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return manager.list()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job.to_public()


@app.get("/api/jobs/{job_id}/metrics")
def get_metrics(job_id: str) -> JSONResponse:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "succeeded":
        raise HTTPException(409, f"job is {job.status}")
    return JSONResponse(read_metrics(job.job_dir))


@app.get("/api/jobs/{job_id}/points")
def get_points(job_id: str, max_points: int | None = None) -> JSONResponse:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "succeeded":
        raise HTTPException(409, f"job is {job.status}")
    kwargs = {"max_points": max_points} if max_points else {}
    return JSONResponse(read_all_clusters(job.job_dir, **kwargs))


@app.get("/api/jobs/{job_id}/voxels")
def get_voxels(job_id: str, voxel_size: float = 0.1) -> JSONResponse:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != "succeeded":
        raise HTTPException(409, f"job is {job.status}")
    try:
        return JSONResponse(voxelise_job(job.job_dir, voxel_size))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
