export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface Job {
  id: string;
  status: JobStatus;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  log: string;
  error: string | null;
  has_results: boolean;
}

export interface Cluster {
  row_id: number | null;
  file: string;
  count: number;
  original_count: number;
  xyz: number[];
  ndvi: number[] | null;
}

export interface PointsPayload {
  center: [number, number, number];
  clusters: Cluster[];
}

export type MetricRow = Record<string, number | string | null>;

export interface VoxelCluster {
  row_id: number | null;
  file: string;
  n_voxels: number;
  n_voxels_sent: number;
  vol_voxel: number;
  n_points: number;
  truncated: boolean;
  xyz: number[];
}

export interface VoxelsPayload {
  voxel_size: number;
  center: [number, number, number];
  clusters: VoxelCluster[];
  total_voxels: number;
  total_volume: number;
  min_voxel_size: number;
  max_voxel_size: number;
  exceeded_total_cap: boolean;
}
