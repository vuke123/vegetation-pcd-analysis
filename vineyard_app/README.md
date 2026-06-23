# Vineyard App

A small web application that visualises the output of the
[vegetation-pcd-analysis](../vegetation-pcd-analysis) pipeline.

Upload a `.las` / `.laz` multispectral point cloud of a vineyard field, the
backend runs the full pipeline (SMRF ground removal → PCL Euclidean clustering
→ per-cluster NDVI → per-row feature extraction), and the frontend renders the
clustered rows in an interactive 3D viewer with labels and metric tables.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Browser (React + Three.js)                                         │
│    upload .las  ─────────────►  POST /api/jobs                      │
│    poll status  ◄─────────────  GET  /api/jobs/{id}                 │
│    fetch 3D    ◄─────────────  GET  /api/jobs/{id}/points           │
│    fetch table ◄─────────────  GET  /api/jobs/{id}/metrics          │
└───────────────┬─────────────────────────────────────────────────────┘
                │ FastAPI (uvicorn)
                ▼
       run_pipeline.sh  ──►  ground classification (Python + PDAL)
                         ──►  clustering_only (C++ / PCL)
                         ──►  pcd_to_ndvi_las.py
                         ──►  compute_row_features.py → row_features.parquet
                         ──►  artifacts copied to data/jobs/<id>/clusters/
```

Nothing in `vegetation-pcd-analysis/` is modified. The upstream pipeline's
natural output dirs (`out_ground/`, `out_cluster/`, `out_cluster_las/`) are
used as a staging area by the pipeline itself; the app copies results out to
`vineyard_app/data/jobs/<id>/` after each run and serialises jobs with a
process lock so two runs never collide.

## Repository layout

```
vineyard_app/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI routes
│   │   ├── pipeline.py     # Job manager — serial subprocess runner
│   │   ├── data.py         # Parquet + LAS → JSON for the UI
│   │   └── config.py       # Paths & limits (env-overridable)
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api.ts
│   │   ├── components/
│   │   │   ├── Uploader.tsx
│   │   │   ├── JobStatus.tsx    # streaming-ish log (1s poll)
│   │   │   ├── Viewer.tsx       # react-three-fiber 3D scene
│   │   │   └── MetricsPanel.tsx # per-row features table
│   │   └── …
│   └── package.json
├── data/                        # per-job working directories (gitignored)
└── README.md
```

## Prerequisites

The backend shells out to the existing pipeline, so the same toolchain must
be available on the host:

- Python 3.10+ with the pipeline's dependencies (PDAL python bindings,
  `open3d`, `laspy`, `numpy`, `pandas`, `scipy`, `shapely`, …). All are
  listed in `backend/requirements.txt` — the backend runs in the same
  environment as the pipeline, so a single `pip install -r` covers both.
- `pdal` CLI
- `cmake`, a C++14 toolchain, PCL ≥ 1.8 (for the `clustering_only` binary —
  the pipeline builds it on the first run)
- Node.js 18+ and npm (frontend)

Confirm the pipeline works standalone first:

```bash
cd ../vegetation-pcd-analysis/scripts
bash run_pipeline.sh /path/to/some.las
```

If that completes and writes `out_cluster_las/row_features.parquet`, the app
will work.

## Setup

### Backend

```bash
cd vineyard_app/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Environment overrides (all optional):

| Variable                          | Default                                                              |
| --------------------------------- | -------------------------------------------------------------------- |
| `VINEYARD_PIPELINE_DIR`           | `<repo>/vegetation-pcd-analysis/scripts`                             |
| `VINEYARD_DATA_DIR`               | `vineyard_app/data`                                                  |
| `VINEYARD_MAX_UPLOAD_BYTES`       | `4294967296` (4 GiB)                                                 |
| `VINEYARD_MAX_POINTS_PER_CLUSTER` | `8000` (downsampling cap for the 3D viewer)                          |

### Frontend

```bash
cd vineyard_app/frontend
npm install
npm run dev
```

Vite runs on <http://localhost:5173> and proxies `/api/*` to the backend on
port 8000, so you just open the Vite URL in a browser.

## API reference

| Method | Path                         | Description                                           |
| ------ | ---------------------------- | ----------------------------------------------------- |
| POST   | `/api/jobs`                  | Multipart upload (`file=` .las/.laz), starts a job    |
| GET    | `/api/jobs`                  | List jobs                                             |
| GET    | `/api/jobs/{id}`             | Status + accumulated pipeline log                     |
| GET    | `/api/jobs/{id}/metrics`     | `row_features.parquet` as a JSON array                |
| GET    | `/api/jobs/{id}/points`      | Downsampled per-cluster xyz (centered), colors client-side |

All endpoints return 404 for unknown jobs and 409 if results are requested
before the job has succeeded.

## How it works

1. **Upload** — Multipart body is streamed to a temp file (bounded by
   `VINEYARD_MAX_UPLOAD_BYTES`) and then moved into
   `data/jobs/<id>/input.las`.
2. **Run** — `JobManager` acquires a process-wide lock, `Popen`s
   `bash run_pipeline.sh <input.las>` with `cwd=scripts/`, and appends stdout
   to the job's log buffer line-by-line. The frontend polls `/api/jobs/{id}`
   once per second to stream the log into the UI.
3. **Collect** — On success, `row_features.parquet` and each
   `*_cluster_*_ndvi.las` are copied from `scripts/out_cluster_las/` into
   `data/jobs/<id>/clusters/`. The scripts' own output dirs are then free to
   be wiped by the next run.
4. **Serve** — On result requests, the backend reads the parquet into JSON
   and loads each cluster LAS with `laspy`, downsamples via fixed stride to
   `VINEYARD_MAX_POINTS_PER_CLUSTER`, and re-centers coordinates to the ground
   centroid so the frontend doesn't have to handle UTM-scale offsets.
5. **Render** — `@react-three/fiber` draws one `THREE.Points` per cluster,
   colored from a golden-ratio hue palette; clicking a cluster (or an entry
   in the right-hand table) selects it and dims the rest. Row-id labels
   float above each cluster via `drei/Html`.

## Concurrency note

Because the upstream pipeline writes to fixed output directories inside
`vegetation-pcd-analysis/scripts/`, `JobManager` runs **one pipeline at a
time** (global `threading.Lock`). Additional uploads are queued and picked
up in order. This is intentional — removing the lock would corrupt results.

## Troubleshooting

- **"pipeline exited with status N"** — reproduce the failure by running
  `run_pipeline.sh` directly; the same log is visible in the UI.
- **Viewer is empty but metrics show up** — the cluster LAS files weren't
  copied out. Check that `*_cluster_*_ndvi.las` exist in the job directory.
- **CMake / PCL errors on first run** — the pipeline builds `clustering_only`
  on step [2/6]; install PCL dev packages and a working `cmake`.

## License

Matches the parent project.
