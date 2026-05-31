import { Canvas } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { VoxelCluster, VoxelsPayload } from "../types";
import { clusterColor } from "../colors";

interface Props {
  data: VoxelsPayload;
  selectedRowId: number | null;
  onSelect: (rowId: number | null) => void;
}

/**
 * Voxelised scene: each occupied voxel is drawn as an axis-aligned cube of
 * edge length data.voxel_size, positioned at the voxel centre (in the same
 * recentred coordinate system as <Viewer />, so the orbit camera framing
 * matches the point-cloud view).
 *
 * One <InstancedMesh> per cluster — keyed on (cluster, voxel_size) so React
 * fully re-mounts the mesh when the slider changes the cube edge, which is
 * the simplest correct way to refresh BoxGeometry args.
 */
export function VoxelView({ data, selectedRowId, onSelect }: Props) {
  const bounds = useMemo(() => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const c of data.clusters) {
      const xyz = c.xyz;
      for (let p = 0; p < xyz.length; p += 3) {
        const rx = xyz[p];
        const ry = xyz[p + 2];     // remap z -> y (Three is Y-up, data is Z-up)
        const rz = -xyz[p + 1];    // remap -y -> z
        if (rx < minX) minX = rx; if (rx > maxX) maxX = rx;
        if (ry < minY) minY = ry; if (ry > maxY) maxY = ry;
        if (rz < minZ) minZ = rz; if (rz > maxZ) maxZ = rz;
      }
    }
    if (!isFinite(minX)) {
      return { minX: -1, maxX: 1, minY: -1, maxY: 1, minZ: -1, maxZ: 1 };
    }
    return { minX, maxX, minY, maxY, minZ, maxZ };
  }, [data]);

  const span = Math.max(
    bounds.maxX - bounds.minX,
    bounds.maxY - bounds.minY,
    bounds.maxZ - bounds.minZ,
    1,
  );
  const cameraPos: [number, number, number] = [span * 0.6, span * 0.7, span * 1.2];

  return (
    <Canvas
      camera={{ position: cameraPos, fov: 45, near: 0.1, far: span * 20 }}
      onPointerMissed={() => onSelect(null)}
    >
      <color attach="background" args={["#070809"]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[10, 20, 10]} intensity={0.8} />
      <directionalLight position={[-12, 10, -8]} intensity={0.25} />

      {data.clusters.map((c, i) => (
        <ClusterVoxels
          key={`${c.file}-${data.voxel_size.toFixed(4)}`}
          cluster={c}
          colorIndex={i}
          voxelSize={data.voxel_size}
          selected={selectedRowId != null && c.row_id === selectedRowId}
          dimmed={selectedRowId != null && c.row_id !== selectedRowId}
          onPick={() => c.row_id != null && onSelect(c.row_id)}
        />
      ))}

      <gridHelper args={[span * 2, 20, "#1a2433", "#0f1520"]} position={[0, bounds.minY, 0]} />
      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
      <GizmoHelper alignment="bottom-right" margin={[60, 60]}>
        <GizmoViewport axisColors={["#ff6b6b", "#58d68d", "#4ea1ff"]} labelColor="white" />
      </GizmoHelper>
    </Canvas>
  );
}

interface ClusterVoxelsProps {
  cluster: VoxelCluster;
  colorIndex: number;
  voxelSize: number;
  selected: boolean;
  dimmed: boolean;
  onPick: () => void;
}

function ClusterVoxels({ cluster, colorIndex, voxelSize, selected, dimmed, onPick }: ClusterVoxelsProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const count = Math.floor(cluster.xyz.length / 3);

  const color = useMemo(() => {
    const [r, g, b] = clusterColor(colorIndex);
    return new THREE.Color(r, g, b);
  }, [colorIndex]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const mat = new THREE.Matrix4();
    const xyz = cluster.xyz;
    for (let p = 0; p < count; p++) {
      const x = xyz[p * 3];
      const y = xyz[p * 3 + 1];
      const z = xyz[p * 3 + 2];
      // Same axis remap as <Viewer />: (x, y, z) -> (x, z, -y)
      mat.identity().setPosition(x, z, -y);
      mesh.setMatrixAt(p, mat);
    }
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
  }, [cluster, count]);

  if (count === 0) return null;

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, count]}
      onClick={(e) => { e.stopPropagation(); onPick(); }}
    >
      <boxGeometry args={[voxelSize, voxelSize, voxelSize]} />
      <meshLambertMaterial
        color={color}
        transparent={dimmed}
        opacity={dimmed ? 0.18 : 1}
        emissive={selected ? color : new THREE.Color(0, 0, 0)}
        emissiveIntensity={selected ? 0.35 : 0}
      />
    </instancedMesh>
  );
}
