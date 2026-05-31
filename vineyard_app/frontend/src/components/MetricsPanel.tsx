import { Fragment } from "react";
import type { MetricRow } from "../types";

interface Props {
  metrics: MetricRow[];
  selectedRowId: number | null;
  onSelect: (rowId: number | null) => void;
}

const HIGHLIGHTED: { key: string; label: string; unit?: string; digits?: number }[] = [
  { key: "point_count", label: "Points", digits: 0 },
  { key: "row_length", label: "Length", unit: "m", digits: 2 },
  { key: "row_width", label: "Width", unit: "m", digits: 2 },
  { key: "height_max", label: "Height max", unit: "m", digits: 2 },
  { key: "height_p90", label: "Height p90", unit: "m", digits: 2 },
  { key: "vol_voxel", label: "Volume (voxel)", unit: "m³", digits: 2 },
  { key: "vol_slice", label: "Volume (slice)", unit: "m³", digits: 2 },
  { key: "ndvi_mean", label: "NDVI mean", digits: 3 },
  { key: "ndvi_p10", label: "NDVI p10", digits: 3 },
  { key: "ndvi_p90", label: "NDVI p90", digits: 3 },
];

function fmt(v: unknown, digits = 2): string {
  if (v == null) return "—";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
}

export function MetricsPanel({ metrics, selectedRowId, onSelect }: Props) {
  if (!metrics.length) {
    return <div className="empty">No metrics yet</div>;
  }

  const byRow = new Map<number, MetricRow>();
  for (const r of metrics) {
    const id = r["row_id"];
    if (typeof id === "number") byRow.set(id, r);
  }
  const selected = selectedRowId != null ? byRow.get(selectedRowId) : null;

  return (
    <div>
      <ul className="row-list">
        {metrics.map((r) => {
          const id = r["row_id"];
          const isActive = typeof id === "number" && id === selectedRowId;
          const displayId = typeof id === "number" ? String(id).padStart(2, "0") : "??";
          return (
            <li
              key={String(id ?? r["cluster_file"])}
              className={`row-card${isActive ? " active" : ""}`}
              onClick={() => (typeof id === "number" ? onSelect(id) : null)}
            >
              <div className="head">
                <span className="label">Row</span>
                <span className="title">{displayId}</span>
              </div>
              <div className="meta">
                <span>{fmt(r["row_length"], 1)} m</span>
                <span>NDVI {fmt(r["ndvi_mean"], 2)}</span>
                <span>{fmt(r["point_count"], 0)} pts</span>
              </div>
              {isActive && (
                <div className="detail-grid">
                  {HIGHLIGHTED.map(({ key, label, unit, digits }) => (
                    <Fragment key={key}>
                      <span className="k">{label}</span>
                      <span className="v">
                        {fmt(r[key], digits)}
                        {unit ? ` ${unit}` : ""}
                      </span>
                    </Fragment>
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ul>
      {selected && (
        <>
          <div className="status-row" style={{ marginTop: 24 }}>
            <span>All metrics · row {selected["row_id"]}</span>
          </div>
          <div className="detail-grid" style={{ marginTop: 0, paddingTop: 0, borderTop: "none" }}>
            {Object.entries(selected).map(([k, v]) => (
              <Fragment key={k}>
                <span className="k">{k}</span>
                <span className="v">{fmt(v, 3)}</span>
              </Fragment>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
