import { Canvas } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport, Html } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";
import type { PointsPayload } from "../types";
import { clusterColor } from "../colors";

interface Props {
  data: PointsPayload;
  selectedRowId: number | null;
  onSelect: (rowId: number | null) => void;
}

interface ClusterMeta {
  rowId: number | null;
  centroid: [number, number, number];
  top: [number, number, number];
  color: [number, number, number];
  count: number;
}

export function Viewer({ data, selectedRowId, onSelect }: Props) {
  const { geometries, metas, bounds } = useMemo(() => {
    const metas: ClusterMeta[] = [];
    const geometries: { geom: THREE.BufferGeometry; meta: ClusterMeta }[] = [];
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;

    data.clusters.forEach((c, i) => {
      const arr = new Float32Array(c.xyz);
      // Three.js is Y-up; source data is Z-up. Remap (x, y, z) -> (x, z, -y).
      const n = arr.length / 3;
      const remapped = new Float32Array(arr.length);
      let cx = 0, cy = 0, cz = 0, maxYval = -Infinity, topX = 0, topZ = 0;
      for (let p = 0; p < n; p++) {
        const x = arr[p * 3];
        const y = arr[p * 3 + 1];
        const z = arr[p * 3 + 2];
        const rx = x;
        const ry = z;
        const rz = -y;
        remapped[p * 3] = rx;
        remapped[p * 3 + 1] = ry;
        remapped[p * 3 + 2] = rz;
        cx += rx; cy += ry; cz += rz;
        if (ry > maxYval) { maxYval = ry; topX = rx; topZ = rz; }
        if (rx < minX) minX = rx; if (rx > maxX) maxX = rx;
        if (ry < minY) minY = ry; if (ry > maxY) maxY = ry;
        if (rz < minZ) minZ = rz; if (rz > maxZ) maxZ = rz;
      }
      cx /= n; cy /= n; cz /= n;

      const color = clusterColor(i);
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.BufferAttribute(remapped, 3));
      const meta: ClusterMeta = {
        rowId: c.row_id,
        centroid: [cx, cy, cz],
        top: [topX, maxYval, topZ],
        color,
        count: c.count,
      };
      metas.push(meta);
      geometries.push({ geom, meta });
    });

    const bounds = { minX, maxX, minY, maxY, minZ, maxZ };
    return { geometries, metas, bounds };
  }, [data]);

  const span = Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    bounds.maxZ - bounds.minZ,
    1
  );
  const cameraPos: [number, number, number] = [span * 0.6, span * 0.7, span * 1.2];

  return (
    <Canvas
      camera={{ position: cameraPos, fov: 45, near: 0.1, far: span * 20 }}
      onPointerMissed={() => onSelect(null)}
    >
      <color attach="background" args={["#070809"]} />
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 20, 10]} intensity={0.6} />

      {geometries.map(({ geom, meta }, i) => {
        const selected = selectedRowId != null && meta.rowId === selectedRowId;
        const dim = selectedRowId != null && !selected;
        const [r, g, b] = meta.color;
        return (
          <points
            key={i}
            geometry={geom}
            onClick={(e) => {
              e.stopPropagation();
              if (meta.rowId != null) onSelect(meta.rowId);
            }}
          >
            <pointsMaterial
              size={selected ? 0.07 : 0.045}
              color={new THREE.Color(r, g, b)}
              sizeAttenuation
              transparent
              opacity={dim ? 0.25 : 1}
            />
          </points>
        );
      })}

      {metas.map((m, i) => (
        m.rowId != null ? (
          <Html
            key={i}
            position={m.top}
            center
            distanceFactor={span * 0.35}
            style={{ pointerEvents: "none" }}
          >
            <div
              style={{
                background: "rgba(8, 10, 16, 0.78)",
                color: selectedRowId === m.rowId ? "#7ab8ff" : "#c3cbd6",
                border: `1px solid ${selectedRowId === m.rowId ? "rgba(122,184,255,0.6)" : "rgba(255,255,255,0.12)"}`,
                padding: "3px 9px",
                borderRadius: 2,
                fontSize: 10,
                fontFamily: "'Space Grotesk', sans-serif",
                fontWeight: 500,
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                whiteSpace: "nowrap",
                backdropFilter: "blur(6px)",
              }}
            >
              Row {String(m.rowId).padStart(2, "0")}
            </div>
          </Html>
        ) : null
      ))}

      <gridHelper args={[span * 2, 20, "#1a2433", "#0f1520"]} position={[0, bounds.minY, 0]} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={["#ff6b6b", "#58d68d", "#4ea1ff"]} labelColor="white" />
      </GizmoHelper>
    </Canvas>
  );
}
