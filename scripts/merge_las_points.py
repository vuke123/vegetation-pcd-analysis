#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import laspy


def hash_xyz_int(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    X = X.astype(np.int64, copy=False)
    Y = Y.astype(np.int64, copy=False)
    Z = Z.astype(np.int64, copy=False)
    return (X * 73856093) ^ (Y * 19349663) ^ (Z * 83492791)


def merge_las_files(
    inputs: list[Path],
    output: Path,
    dedup: bool = True,
    chunk_size: int = 1_000_000,
) -> None:
    if not inputs:
        raise ValueError("No input LAS files provided.")

    output.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(inputs[0])) as r0:
        header = r0.header.copy()
        sig0 = tuple(r0.header.point_format.dimension_names)

    print(f"Template: {inputs[0]}")
    print(f"Dimensions ({len(sig0)}): {sig0}")

    seen_hashes: set[int] | None = set() if dedup else None
    total_in = 0
    total_out = 0

    with laspy.open(str(output), mode="w", header=header) as writer:
        for i, f in enumerate(inputs):
            print(f"[{i+1}/{len(inputs)}] Reading: {f}")

            with laspy.open(str(f)) as reader:
                sig = tuple(reader.header.point_format.dimension_names)
                if sig != sig0:
                    raise RuntimeError(
                        f"Dimension mismatch in {f}\n"
                        f"Expected: {sig0}\n"
                        f"Found:    {sig}\n"
                        f"Fix: ensure all clusters are exported with the SAME template/header and extra dims."
                    )

                for chunk in reader.chunk_iterator(chunk_size):
                    n = len(chunk)
                    total_in += n
                    if n == 0:
                        continue

                    if not dedup:
                        writer.write_points(chunk.points)
                        total_out += n
                        continue

                    h = hash_xyz_int(chunk.X, chunk.Y, chunk.Z)
                    mask_keep = np.ones(n, dtype=bool)

                    for idx, hv in enumerate(h):
                        hv_int = int(hv)
                        if hv_int in seen_hashes:
                            mask_keep[idx] = False
                        else:
                            seen_hashes.add(hv_int)

                    if np.any(mask_keep):
                        writer.write_points(chunk.points[mask_keep])
                        total_out += int(np.count_nonzero(mask_keep))

    print(f"\n✅ Wrote merged LAS: {output}")
    print(f"Total input points:  {total_in:,}")
    print(f"Total output points: {total_out:,}")
    if dedup:
        print(f"Removed duplicates:  {total_in - total_out:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--pattern", default="*_ndvi.las")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out = Path(args.out)

    files = sorted(in_dir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No files matched {in_dir}/{args.pattern}")

    out_resolved = out.resolve()
    files = [f for f in files if f.resolve() != out_resolved]
    files = [f for f in files if f.stat().st_size > 0]

    if not files:
        raise SystemExit("After filtering, no non-empty input LAS files remain.")

    merge_las_files(
        inputs=files,
        output=out,
        dedup=not args.no_dedup,
        chunk_size=args.chunk,
    )


if __name__ == "__main__":
    main()
