import { useEffect, useRef, useState } from "react";
import { getJob, getMetrics, getPoints, uploadLas } from "./api";
import { Viewer } from "./components/Viewer";
import { MetricsPanel } from "./components/MetricsPanel";
import { VoxelPage } from "./components/VoxelPage";
import type { Job, MetricRow, PointsPayload } from "./types";

const NAV = [
  { key: "overview", label: "Overview" },
  { key: "upload", label: "Upload" },
  { key: "pipeline", label: "Pipeline" },
  { key: "results", label: "Results" },
  { key: "voxelize", label: "Voxelize" },
];

type View = "results" | "voxelize";

export default function App() {
  const [job, setJob] = useState<Job | null>(null);
  const [metrics, setMetrics] = useState<MetricRow[]>([]);
  const [points, setPoints] = useState<PointsPayload | null>(null);
  const [selectedRowId, setSelectedRowId] = useState<number | null>(null);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [uploadErr, setUploadErr] = useState<string | null>(null);
  const [view, setView] = useState<View>("results");
  const resultsLoaded = useRef<string | null>(null);

  useEffect(() => {
    if (!job || job.status === "succeeded" || job.status === "failed") return;
    const iv = setInterval(async () => {
      try {
        setJob(await getJob(job.id));
      } catch (e) { console.error(e); }
    }, 1000);
    return () => clearInterval(iv);
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (!job || job.status !== "succeeded") return;
    if (resultsLoaded.current === job.id) return;
    resultsLoaded.current = job.id;
    setResultsLoading(true);
    Promise.all([getMetrics(job.id), getPoints(job.id)])
      .then(([m, p]) => { setMetrics(m); setPoints(p); })
      .catch((e) => console.error(e))
      .finally(() => setResultsLoading(false));
  }, [job?.id, job?.status]);

  const handleFile = async (file: File | null) => {
    if (!file) return;
    setUploadErr(null);
    try {
      const j = await uploadLas(file);
      setJob(j);
      setMetrics([]);
      setPoints(null);
      setSelectedRowId(null);
      resultsLoaded.current = null;
    } catch (e) {
      setUploadErr((e as Error).message);
    }
  };

  const startNew = () => {
    setJob(null);
    setMetrics([]);
    setPoints(null);
    setSelectedRowId(null);
    setUploadErr(null);
    setView("results");
    resultsLoaded.current = null;
  };

  const busy = !!job && (job.status === "queued" || job.status === "running");
  const canVoxelize = !!job && job.status === "succeeded";
  const activeNav = !job
    ? "overview"
    : busy
    ? "pipeline"
    : job.status === "succeeded"
    ? view === "voxelize"
      ? "voxelize"
      : "results"
    : "pipeline";

  return (
    <>
      <div className="scene">
        <div className="grain" />
        <div className="vignette" />
      </div>

      <div className="brand">
        <span className="dot" />
        <span>Vitis · PCD Intelligence</span>
      </div>

      <nav className="nav" aria-label="sections">
        {NAV.map((n, i) => {
          const handler =
            n.key === "overview"
              ? startNew
              : n.key === "voxelize" && canVoxelize
              ? () => setView("voxelize")
              : n.key === "results" && canVoxelize && view === "voxelize"
              ? () => setView("results")
              : undefined;
          const disabled = n.key === "voxelize" && !canVoxelize;
          return (
            <a
              key={n.key}
              className={activeNav === n.key ? "active" : ""}
              onClick={handler}
              style={{
                cursor: handler ? "pointer" : "default",
                opacity: disabled ? 0.35 : undefined,
              }}
            >
              <span className="num">{String(i + 1).padStart(2, "0")}</span>
              <span className="line" />
              <span>{n.label}</span>
            </a>
          );
        })}
      </nav>

      {job && (
        <div className="status-chip">
          <span>Job {job.id.slice(0, 6)}</span>
          <span className={`badge ${job.status}`}>{job.status}</span>
        </div>
      )}

      <div className="root">
        {!job ? (
          <Landing onFile={handleFile} error={uploadErr} />
        ) : view === "voxelize" && canVoxelize ? (
          <VoxelPage
            job={job}
            selectedRowId={selectedRowId}
            onSelect={setSelectedRowId}
            onBack={() => setView("results")}
          />
        ) : (
          <Workspace
            job={job}
            metrics={metrics}
            points={points}
            selectedRowId={selectedRowId}
            onSelect={setSelectedRowId}
            onNew={startNew}
            onFile={handleFile}
            uploadErr={uploadErr}
            resultsLoading={resultsLoading}
            busy={busy}
          />
        )}
      </div>
    </>
  );
}

function Landing({ onFile, error }: { onFile: (f: File | null) => void; error: string | null }) {
  return (
    <section className="hero">
      <div className="eyebrow">Precision Viticulture · Multispectral LiDAR</div>
      <h1>
        Vineyard
        <span className="accent">intelligence</span>
        at canopy scale
      </h1>
      <p className="lede">
        Upload a multispectral point cloud and watch ground classification, row clustering,
        NDVI synthesis and per-row volumetric metrics resolve into a single, navigable
        3D scene — rendered in the browser.
      </p>
      <div className="cta-row">
        <label className="cta">
          <input type="file" accept=".las,.laz" onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
          <span>Initiate analysis</span>
          <span className="arrow">→</span>
        </label>
      </div>
      {error && <div className="file-error" style={{ marginTop: 24 }}>{error}</div>}

      <div className="hero-meta">
        <div className="item"><b>SMRF</b>Ground removal</div>
        <div className="item"><b>PCL · C++</b>Row clustering</div>
        <div className="item"><b>NDVI</b>Multispectral</div>
        <div className="item"><b>Parquet</b>Row features</div>
      </div>
    </section>
  );
}

interface WsProps {
  job: Job;
  metrics: MetricRow[];
  points: PointsPayload | null;
  selectedRowId: number | null;
  onSelect: (id: number | null) => void;
  onNew: () => void;
  onFile: (f: File | null) => void;
  uploadErr: string | null;
  resultsLoading: boolean;
  busy: boolean;
}

function Workspace(p: WsProps) {
  const elapsed =
    p.job.started_at != null
      ? ((p.job.finished_at ?? Date.now() / 1000) - p.job.started_at).toFixed(1)
      : null;
  const totalPts = p.points ? p.points.clusters.reduce((a, c) => a + c.count, 0) : 0;

  return (
    <div className="workspace">
      <aside className="pane">
        <header>
          <h2>Job</h2>
          <button className="cta ghost" style={{ padding: "6px 12px", fontSize: 9 }} onClick={p.onNew} disabled={p.busy}>
            New
          </button>
        </header>
        <div className="pane-body">
          <div className="status-row">
            <span className="id">{p.job.id.slice(0, 10)}</span>
            <span>{elapsed ? `${elapsed}s` : "—"}</span>
          </div>

          <label className="uploader-inline" style={{ opacity: p.busy ? 0.4 : 1, pointerEvents: p.busy ? "none" : "auto" }}>
            <input type="file" accept=".las,.laz" onChange={(e) => p.onFile(e.target.files?.[0] ?? null)} />
            <div className="label">{p.busy ? "Pipeline running" : "Replace input"}</div>
            <div className="hint">.las · .laz · multispectral</div>
          </label>
          {p.uploadErr && <div className="file-error">{p.uploadErr}</div>}

          {p.job.error && (
            <div className="file-error" style={{ marginTop: 14 }}>{p.job.error}</div>
          )}

          <div style={{ marginTop: 18 }}>
            <div className="status-row"><span>Log</span></div>
            <pre className="log">{p.job.log || "…"}</pre>
          </div>
        </div>
      </aside>

      <section className="viewer-pane">
        {p.points ? (
          <>
            <div className="hud">
              <div className="stat"><span className="k">Clusters</span><span className="v">{p.points.clusters.length}</span></div>
              <div className="divider" />
              <div className="stat"><span className="k">Points</span><span className="v">{totalPts.toLocaleString()}</span></div>
              {p.selectedRowId != null && (
                <>
                  <div className="divider" />
                  <div className="stat"><span className="k">Selected</span><span className="v">Row {p.selectedRowId}</span></div>
                </>
              )}
            </div>
            <Viewer data={p.points} selectedRowId={p.selectedRowId} onSelect={p.onSelect} />
          </>
        ) : (
          <div className="empty" style={{ paddingTop: 160 }}>
            {p.resultsLoading
              ? "Loading scene…"
              : p.job.status === "failed"
              ? "Pipeline failed — see log"
              : "Processing point cloud…"}
          </div>
        )}
      </section>

      <aside className="pane">
        <header>
          <h2>Row features</h2>
          <span className="count">{p.metrics.length ? `${p.metrics.length.toString().padStart(2, "0")}` : ""}</span>
        </header>
        <div className="pane-body">
          <MetricsPanel metrics={p.metrics} selectedRowId={p.selectedRowId} onSelect={p.onSelect} />
        </div>
      </aside>
    </div>
  );
}
