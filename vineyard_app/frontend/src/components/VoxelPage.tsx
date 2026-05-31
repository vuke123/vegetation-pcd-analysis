import { useEffect, useMemo, useRef, useState } from "react";
import { getVoxels } from "../api";
import { VoxelView } from "./VoxelView";
import type { Job, VoxelsPayload } from "../types";

interface Props {
  job: Job;
  selectedRowId: number | null;
  onSelect: (rowId: number | null) => void;
  onBack: () => void;
}

// Log-scaled slider over voxel size in metres.
// The backend clamps to [MIN_VOXEL_SIZE, MAX_VOXEL_SIZE] = [0.02, 1.00];
// we deliberately stop the UI at 0.50 m which is the point past which the
// scene degenerates into a handful of axis-aligned blocks.
const SLIDER_MIN_M = 0.02;
const SLIDER_MAX_M = 0.50;
const SLIDER_STEPS = 240;
const FETCH_DEBOUNCE_MS = 220;

function sliderToVoxel(s: number): number {
  const t = Math.min(1, Math.max(0, s / SLIDER_STEPS));
  return SLIDER_MIN_M * Math.pow(SLIDER_MAX_M / SLIDER_MIN_M, t);
}

function voxelToSlider(v: number): number {
  const t = Math.log(v / SLIDER_MIN_M) / Math.log(SLIDER_MAX_M / SLIDER_MIN_M);
  return Math.round(Math.min(1, Math.max(0, t)) * SLIDER_STEPS);
}

export function VoxelPage({ job, selectedRowId, onSelect, onBack }: Props) {
  const [sliderValue, setSliderValue] = useState<number>(() => voxelToSlider(0.08));
  const voxelSize = useMemo(() => sliderToVoxel(sliderValue), [sliderValue]);

  const [data, setData] = useState<VoxelsPayload | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);
  const requestSeq = useRef(0);
  const fetchTimer = useRef<number | null>(null);

  // Debounced fetch whenever the slider value (=> voxelSize) changes.
  useEffect(() => {
    if (fetchTimer.current != null) {
      window.clearTimeout(fetchTimer.current);
    }
    const seq = ++requestSeq.current;
    setLoading(true);
    setErr(null);
    fetchTimer.current = window.setTimeout(() => {
      getVoxels(job.id, voxelSize)
        .then((payload) => {
          if (seq !== requestSeq.current) return;  // stale, newer slider event in flight
          setData(payload);
          setLoading(false);
        })
        .catch((e) => {
          if (seq !== requestSeq.current) return;
          setErr((e as Error).message);
          setLoading(false);
        });
    }, FETCH_DEBOUNCE_MS);
    return () => {
      if (fetchTimer.current != null) window.clearTimeout(fetchTimer.current);
    };
  }, [job.id, voxelSize]);

  const totalVoxels = data?.total_voxels ?? 0;
  const totalVolume = data?.total_volume ?? 0;
  const clusters = data?.clusters ?? [];
  const truncated = clusters.some((c) => c.truncated) || (data?.exceeded_total_cap ?? false);

  return (
    <div className="workspace">
      <aside className="pane">
        <header>
          <h2>Voxelize</h2>
          <button
            className="cta ghost"
            style={{ padding: "6px 12px", fontSize: 9 }}
            onClick={onBack}
          >
            Back
          </button>
        </header>
        <div className="pane-body">
          <div className="status-row">
            <span>Voxel size</span>
            <span className="v" style={{ color: "#7ab8ff", fontVariantNumeric: "tabular-nums" }}>
              {voxelSize.toFixed(3)} m
            </span>
          </div>

          <input
            type="range"
            min={0}
            max={SLIDER_STEPS}
            step={1}
            value={sliderValue}
            onChange={(e) => setSliderValue(parseInt(e.target.value, 10))}
            style={{ width: "100%", marginTop: 8, accentColor: "#7ab8ff" }}
            aria-label="Voxel size"
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, opacity: 0.55, marginTop: 4, letterSpacing: "0.15em" }}>
            <span>{SLIDER_MIN_M.toFixed(2)} m · fine</span>
            <span>{SLIDER_MAX_M.toFixed(2)} m · coarse</span>
          </div>

          <div style={{ marginTop: 22, display: "grid", gridTemplateColumns: "1fr auto", rowGap: 6, fontSize: 11 }}>
            <span style={{ opacity: 0.6 }}>occupied voxels</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>{totalVoxels.toLocaleString()}</span>
            <span style={{ opacity: 0.6 }}>total volume</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>{totalVolume.toFixed(2)} m³</span>
            <span style={{ opacity: 0.6 }}>cell volume</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>{(voxelSize ** 3).toExponential(2)} m³</span>
            <span style={{ opacity: 0.6 }}>clusters</span>
            <span style={{ fontVariantNumeric: "tabular-nums" }}>{clusters.length}</span>
          </div>

          {truncated && data && (
            <div className="file-error" style={{ marginTop: 14, fontSize: 10 }}>
              Cube count exceeded render budget — a random subset is shown.
              Volume value above is still the true voxel count × s³.
            </div>
          )}
          {err && (
            <div className="file-error" style={{ marginTop: 14 }}>{err}</div>
          )}

          <div style={{ marginTop: 22 }}>
            <div className="status-row"><span>Rows</span></div>
            <ul style={{ listStyle: "none", padding: 0, margin: 0, marginTop: 6 }}>
              {clusters.map((c) => {
                const id = c.row_id;
                const isActive = id != null && id === selectedRowId;
                return (
                  <li
                    key={c.file}
                    onClick={() => (id != null ? onSelect(isActive ? null : id) : undefined)}
                    style={{
                      cursor: id != null ? "pointer" : "default",
                      padding: "8px 10px",
                      marginBottom: 4,
                      border: `1px solid ${isActive ? "rgba(122,184,255,0.5)" : "rgba(255,255,255,0.06)"}`,
                      background: isActive ? "rgba(122,184,255,0.08)" : "transparent",
                      borderRadius: 2,
                      fontSize: 10,
                      letterSpacing: "0.12em",
                      display: "flex",
                      justifyContent: "space-between",
                    }}
                  >
                    <span>Row {id != null ? String(id).padStart(2, "0") : "??"}</span>
                    <span style={{ opacity: 0.6, fontVariantNumeric: "tabular-nums" }}>
                      {c.n_voxels.toLocaleString()} cells · {c.vol_voxel.toFixed(1)} m³
                    </span>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>
      </aside>

      <section className="viewer-pane">
        <div className="hud">
          <div className="stat"><span className="k">Voxel</span><span className="v">{voxelSize.toFixed(3)} m</span></div>
          <div className="divider" />
          <div className="stat"><span className="k">Cells</span><span className="v">{totalVoxels.toLocaleString()}</span></div>
          <div className="divider" />
          <div className="stat"><span className="k">Volume</span><span className="v">{totalVolume.toFixed(1)} m³</span></div>
          {loading && (
            <>
              <div className="divider" />
              <div className="stat"><span className="k">Status</span><span className="v">computing…</span></div>
            </>
          )}
        </div>
        {data && data.clusters.length > 0 ? (
          <VoxelView data={data} selectedRowId={selectedRowId} onSelect={onSelect} />
        ) : (
          <div className="empty" style={{ paddingTop: 160 }}>
            {loading ? "Voxelizing point cloud…" : err ? "Failed to load voxels" : "No voxels"}
          </div>
        )}
      </section>
    </div>
  );
}
