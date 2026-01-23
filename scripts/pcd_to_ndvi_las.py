#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import laspy

# pip install pypcd4
from pypcd4 import PointCloud


def _pcd_to_dataframe_all_fields(pcd_path: str) -> pd.DataFrame:
    """
    Load a .pcd (ascii or binary) and return a DataFrame containing ALL fields
    present in the PCD. Works for PCL-generated PCDs with many scalar fields.
    """
    p = Path(pcd_path)
    if not p.is_file():
        raise FileNotFoundError(f"PCD not found: {p}")

    print(f"📂 Loading PCD (all fields): {p}")
    pc = PointCloud.from_path(str(p))

    fields = list(pc.fields)  # e.g. ['x','y','z','intensity','red','infrared',...]
    if not fields:
        raise ValueError("PCD has no fields?")

    # Best path: structured array (named columns + proper dtypes) if available
    pc_data = getattr(pc, "pc_data", None)
    if pc_data is not None and getattr(pc_data.dtype, "names", None):
        df = pd.DataFrame({name: pc_data[name] for name in pc_data.dtype.names})
        print("Fields:", list(df.columns))
        print(f"Points: {len(df):,}")
        return df

    # Fallback: plain NxF array
    arr = pc.numpy(fields)
    df = pd.DataFrame(arr, columns=fields)
    print("Fields:", list(df.columns))
    print(f"Points: {len(df):,}")
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Available: {list(df.columns)[:30]}...")


def add_ndvi(df: pd.DataFrame,
             red_col: str = "red",
             nir_col: str = "infrared",
             out_col: str = "ndvi",
             eps: float = 1e-6) -> pd.DataFrame:
    """
    NDVI = (NIR - Red) / (NIR + Red)
    """
    red = df[red_col].astype(np.float32)
    nir = df[nir_col].astype(np.float32)
    df[out_col] = (nir - red) / (nir + red + eps)
    return df


import laspy
import numpy as np
import pandas as pd
from pathlib import Path

def export_filtered_las(df: pd.DataFrame, las_template: laspy.LasData, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # pick xyz column names
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"Missing any of {cands}. Have: {list(df.columns)}")

    x_col = pick(["x", "X"])
    y_col = pick(["y", "Y"])
    z_col = pick(["z", "Z"])

    n = len(df)
    if n == 0:
        raise ValueError("DataFrame is empty")

    # ---- FIX: copy header but allocate output for n points (NOT template point count) ----
    hdr = las_template.header.copy()
    hdr.point_count = n
    # keep it consistent
    hdr.number_of_points_by_return = [0] * len(hdr.number_of_points_by_return)

    out = laspy.LasData(hdr)
    out.points = laspy.ScaleAwarePointRecord.zeros(n, header=hdr)
    # -------------------------------------------------------------------------------

    # set XYZ (laspy will scale/offset using hdr)
    out.x = df[x_col].to_numpy(np.float64, copy=False)
    out.y = df[y_col].to_numpy(np.float64, copy=False)
    out.z = df[z_col].to_numpy(np.float64, copy=False)

    # helper to set a dim or create extra bytes
    dim_names = set(out.point_format.dimension_names)

    def set_or_extra(name: str, values: np.ndarray, dtype):
        lname = name.lower()
        if lname in dim_names:
            setattr(out, lname, values.astype(dtype, copy=False))
            return
        # LAS 1.4 NIR dim in laspy is usually "nir"
        if lname == "infrared" and "nir" in dim_names:
            out.nir = values.astype(dtype, copy=False)
            return
        if lname not in dim_names:
            out.add_extra_dim(laspy.ExtraBytesParams(name=lname, type=dtype))
            dim_names.add(lname)
        setattr(out, lname, values.astype(dtype, copy=False))

    # copy red / infrared and ndvi
    if "red" in df.columns:
        red = np.clip(df["red"].to_numpy(np.float64, copy=False), 0, 65535).astype(np.uint16)
        set_or_extra("red", red, np.uint16)

    if "infrared" in df.columns:
        nir = np.clip(df["infrared"].to_numpy(np.float64, copy=False), 0, 65535).astype(np.uint16)
        set_or_extra("infrared", nir, np.uint16)

    if "ndvi" in df.columns:
        set_or_extra("ndvi", df["ndvi"].to_numpy(np.float32, copy=False), np.float32)

    # optional fields (safe + clipped)
    for col, dtype in [
        ("intensity", np.uint16),
        ("classification", np.uint8),
        ("gpstime", np.float64),
        ("returnnumber", np.uint8),
        ("numberofreturns", np.uint8),
    ]:
        if col in df.columns:
            v = df[col].to_numpy(copy=False)
            if np.issubdtype(dtype, np.integer):
                v = np.clip(v.astype(np.float64, copy=False), 0, np.iinfo(dtype).max).astype(dtype)
            else:
                v = v.astype(dtype, copy=False)
            set_or_extra(col, v, dtype)

    out.write(str(out_path))
    print(f"✅ Wrote LAS: {out_path} (points={n:,})")
    print("✅ Output dims:", list(out.point_format.dimension_names))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True, help="Input cluster PCD path")
    ap.add_argument("--template-las", required=True, help="A LAS/LAZ file to copy header/scale/offset/SRS from")
    ap.add_argument("--out-las", required=True, help="Output LAS path")
    args = ap.parse_args()

    # Read template LAS
    template_path = Path(args.template_las)
    if not template_path.is_file():
        raise FileNotFoundError(f"Template LAS not found: {template_path}")
    las_template = laspy.read(str(template_path))

    # Load PCD -> DF
    df = _pcd_to_dataframe_all_fields(args.pcd)

    # Ensure required columns exist
    if "red" not in df.columns or "infrared" not in df.columns:
        raise KeyError(f"PCD DF must contain 'red' and 'infrared'. Columns: {list(df.columns)}")

    # NDVI
    df = add_ndvi(df, red_col="red", nir_col="infrared", out_col="ndvi")

    # Export LAS
    export_filtered_las(df, las_template, args.out_las)


if __name__ == "__main__":
    main()
