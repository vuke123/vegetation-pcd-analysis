import type { Job, MetricRow, PointsPayload, VoxelsPayload } from "./types";

const BASE = "/api";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function uploadLas(file: File): Promise<Job> {
  const fd = new FormData();
  fd.append("file", file);
  return json<Job>(await fetch(`${BASE}/jobs`, { method: "POST", body: fd }));
}

export async function getJob(id: string): Promise<Job> {
  return json<Job>(await fetch(`${BASE}/jobs/${id}`));
}

export async function getMetrics(id: string): Promise<MetricRow[]> {
  return json<MetricRow[]>(await fetch(`${BASE}/jobs/${id}/metrics`));
}

export async function getPoints(id: string): Promise<PointsPayload> {
  return json<PointsPayload>(await fetch(`${BASE}/jobs/${id}/points`));
}

export async function getVoxels(id: string, voxelSize: number): Promise<VoxelsPayload> {
  const url = `${BASE}/jobs/${id}/voxels?voxel_size=${encodeURIComponent(voxelSize)}`;
  return json<VoxelsPayload>(await fetch(url));
}
