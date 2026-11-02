# -*- coding: utf-8 -*-
"""
opening_location_evaluation_all45.py

Direct opening-location evaluation using nearest-position matching.
This version matches openings by nearest 3D center within the same category
(Window-to-Window, Door-to-Door) and writes a single output CSV.

Output:
- opening_location_error_results.csv

"""

import os
import math
import numpy as np
import pandas as pd


# -------------------------------------------------
# CHANGE THESE PATHS
# -------------------------------------------------
REF_CSV = r"F:\CreateBIM\dataset\opening_validation\reference_opening_centers.csv"
AUTO_CSV = r"F:\CreateBIM\dataset\validation\automatic_opening_centers.csv"
OUT_CSV = r"F:\CreateBIM\dataset\opening_validation\opening_location_error_results.csv"

# Maximum allowed matching distance in meters
MAX_MATCH_DISTANCE_M = 1.50

VALID_CATEGORIES = {"Window", "Door"}


def safe_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError("File not found: {}".format(path))

    df = pd.read_csv(path)

    required_cols = ["category", "x_m", "y_m", "z_m"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in {}: {}".format(path, missing))

    for col in ["x_m", "y_m", "z_m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["category", "family", "type", "mark", "level", "unique_id", "element_id"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_str)
        else:
            df[col] = ""

    df = df.dropna(subset=["x_m", "y_m", "z_m"]).copy()
    df = df[df["category"].isin(VALID_CATEGORIES)].copy()
    df = df.reset_index(drop=True)

    return df


def pairwise_distances(ref_xyz, auto_xyz):
    diff = ref_xyz[:, None, :] - auto_xyz[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def greedy_one_to_one_match(ref_df, auto_df, max_dist):
    """
    One-to-one nearest-position matching within same category.
    Returns list of (ref_idx, auto_idx, distance).
    """
    matches = []

    if len(ref_df) == 0 or len(auto_df) == 0:
        return matches

    ref_xyz = ref_df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    auto_xyz = auto_df[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)

    dist_matrix = pairwise_distances(ref_xyz, auto_xyz)

    candidates = []
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            d = float(dist_matrix[i, j])
            if d <= max_dist:
                candidates.append((d, i, j))

    candidates.sort(key=lambda x: x[0])

    used_ref = set()
    used_auto = set()

    for d, i_local, j_local in candidates:
        ref_idx = ref_df.index[i_local]
        auto_idx = auto_df.index[j_local]

        if ref_idx in used_ref or auto_idx in used_auto:
            continue

        used_ref.add(ref_idx)
        used_auto.add(auto_idx)
        matches.append((ref_idx, auto_idx, d))

    return matches


def build_results(ref_df, auto_df, matches):
    rows = []

    for ref_idx, auto_idx, _ in matches:
        ref_row = ref_df.loc[ref_idx]
        auto_row = auto_df.loc[auto_idx]

        dx = float(auto_row["x_m"] - ref_row["x_m"])
        dy = float(auto_row["y_m"] - ref_row["y_m"])
        dz = float(auto_row["z_m"] - ref_row["z_m"])
        d3 = math.sqrt(dx * dx + dy * dy + dz * dz)

        rows.append({
            "ref_unique_id": ref_row["unique_id"],
            "ref_element_id": ref_row["element_id"],
            "ref_category": ref_row["category"],
            "ref_family": ref_row["family"],
            "ref_type": ref_row["type"],
            "ref_mark": ref_row["mark"],
            "ref_level": ref_row["level"],
            "ref_x_m": ref_row["x_m"],
            "ref_y_m": ref_row["y_m"],
            "ref_z_m": ref_row["z_m"],

            "auto_unique_id": auto_row["unique_id"],
            "auto_element_id": auto_row["element_id"],
            "auto_category": auto_row["category"],
            "auto_family": auto_row["family"],
            "auto_type": auto_row["type"],
            "auto_mark": auto_row["mark"],
            "auto_level": auto_row["level"],
            "auto_x_m": auto_row["x_m"],
            "auto_y_m": auto_row["y_m"],
            "auto_z_m": auto_row["z_m"],

            "dx_m": dx,
            "dy_m": dy,
            "dz_m": dz,
            "abs_dx_m": abs(dx),
            "abs_dy_m": abs(dy),
            "abs_dz_m": abs(dz),
            "distance_3d_m": d3,
            "distance_3d_mm": d3 * 1000.0,
            "match_method": "nearest_position_same_category"
        })

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(by=["ref_category", "distance_3d_m"]).reset_index(drop=True)

    return result_df


def summarize(df):
    if df.empty:
        return None

    d = df["distance_3d_m"].to_numpy(dtype=float)
    dx = df["dx_m"].to_numpy(dtype=float)
    dy = df["dy_m"].to_numpy(dtype=float)
    dz = df["dz_m"].to_numpy(dtype=float)

    return {
        "count": int(len(df)),
        "mean_3d_error_m": float(np.mean(d)),
        "rmse_3d_error_m": float(np.sqrt(np.mean(d ** 2))),
        "median_3d_error_m": float(np.median(d)),
        "p95_3d_error_m": float(np.percentile(d, 95)),
        "max_3d_error_m": float(np.max(d)),
        "mean_abs_dx_m": float(np.mean(np.abs(dx))),
        "mean_abs_dy_m": float(np.mean(np.abs(dy))),
        "mean_abs_dz_m": float(np.mean(np.abs(dz))),
    }


def main():
    ref_df = load_csv(REF_CSV)
    auto_df = load_csv(AUTO_CSV)

    print("Reference openings:", len(ref_df))
    print("Automatic openings:", len(auto_df))

    all_matches = []

    for category in sorted(VALID_CATEGORIES):
        ref_cat = ref_df[ref_df["category"] == category].copy()
        auto_cat = auto_df[auto_df["category"] == category].copy()

        matches = greedy_one_to_one_match(ref_cat, auto_cat, MAX_MATCH_DISTANCE_M)
        all_matches.extend(matches)

        print("\nCategory:", category)
        print("  Reference:", len(ref_cat))
        print("  Automatic:", len(auto_cat))
        print("  Matched by nearest position:", len(matches))

    result_df = build_results(ref_df, auto_df, all_matches)

    out_dir = os.path.dirname(OUT_CSV)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    result_df.to_csv(OUT_CSV, index=False)

    summary = summarize(result_df)

    print("\n=== Opening Location Evaluation Summary ===")
    if summary is None:
        print("No matches found.")
        return

    for k, v in summary.items():
        print("{}: {}".format(k, v))

    print("\nSaved detailed results to:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()