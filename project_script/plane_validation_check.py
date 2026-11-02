import numpy as np

# -----------------------------
# helpers
# -----------------------------
def normalize(v, eps=1e-12):
    """Return (unit_vector, norm)."""
    v = np.asarray(v, dtype=float)
    nrm = float(np.linalg.norm(v))
    if nrm < eps:
        return v * 0.0, nrm
    return v / nrm, nrm


# -----------------------------
# your block (USED AS-IS)
# -----------------------------
def plane_fit_pca_tls(points):
    """Total least squares plane via PCA. Returns (centroid, unit_normal, d) for n·x + d = 0."""
    c = points.mean(axis=0)
    X = points - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1, :]
    n, _ = normalize(n)
    d = -float(np.dot(n, c))
    return c, n, d


# -----------------------------
# distances + stats
# -----------------------------
def point_to_plane_distances(points_xyz, n_unit, d):
    """
    Plane: n·x + d = 0, n_unit must be unit.
    Returns absolute orthogonal distances.
    """
    pts = np.asarray(points_xyz, dtype=float)
    n_unit = np.asarray(n_unit, dtype=float)
    return np.abs(pts @ n_unit + d)  # since n is unit

def tilt_deg_from_vertical(n_unit):
    """Deviation from a perfectly vertical plane (Z-up). 0° = vertical wall, 90° = horizontal."""
    c = float(n_unit[2])
    c = np.clip(abs(c), 0.0, 1.0)
    return float(np.degrees(np.arcsin(c)))


def residual_stats(dist):
    """Compute mean, median, RMSE, and 95th percentile."""
    dist = np.asarray(dist, dtype=float)
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "rmse": float(np.sqrt(np.mean(dist**2))),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


# -----------------------------
# per-cluster validation
# -----------------------------
def validate_plane_fit(points_xyz,
                       ignore_label=-1,
                       min_cluster_size=200):

    arr = np.asarray(points_xyzL)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("points_xyzL must be (N,4+) with columns [x,y,z,label].")

    xyz = arr[:, :3].astype(float)
    labels = arr[:, 3].astype(int)

    results = []

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != ignore_label]

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if idx.size < min_cluster_size:
            continue

        pts = xyz[idx]
        c, n, d = plane_fit_pca_tls(pts)

        dist = point_to_plane_distances(pts, n, d)
        stats = residual_stats(dist)

        tilt_vert = tilt_deg_from_vertical(n)

        results.append({
            "label": int(lab),
            "n_points": int(idx.size),
            "a": float(n[0]), "b": float(n[1]), "c": float(n[2]),
            "d": float(d),

            # residual stats...
            "mean_dist": stats["mean"],
            "median_dist": stats["median"],
            "rmse_dist": stats["rmse"],
            "p95_dist": stats["p95"],
           # "max_dist": stats["max"],
            "tilt_deg_from_vertical": tilt_vert,
        })

    results.sort(key=lambda r: r["label"])
    return results

def print_plane_fit_report(results):
    print("\n[TLS plane-fit validation (PCA/TLS per cluster)]")
    print("label | pts |    normal (a,b,c)     |    d     | mean |  med |   rmse |   p95 | tilt_deg_from_vertical|")
    print("-"*86)
    for r in results:
        print(f"{r['label']:5d} | {r['n_points']:4d} | "
              f"({r['a']:+.4f},{r['b']:+.4f},{r['c']:+.4f}) | "
              f"{r['d']:+.4f} | "
              f"{r['mean_dist']:.4f} |" f" {r['median_dist']:.4f} |" f" {r['rmse_dist']:.4f} |" f" {r['p95_dist']:.4f} |" f" {r['tilt_deg_from_vertical']:.4f}")


def save_plane_fit_csv(results, csv_path, decimals=4):
    """Save results to CSV with fixed digits after '.' for float values."""
    import csv
    if not results:
        print("[WARN] No results to save.")
        return

    keys = list(results[0].keys())

    def fmt(v):
        return f"{v:.{decimals}f}" if isinstance(v, float) else v

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: fmt(r[k]) for k in keys})

    print(f"[INFO] Saved plane-fit report → {csv_path}")


# -----------------------------
# OPTIONAL: normal consistency
# -----------------------------
def normal_consistency(normals_xyz, labels, ignore_label=-1, min_cluster_size=200):

    normals = np.asarray(normals_xyz, dtype=float)
    labels = np.asarray(labels, dtype=int)

    out = []
    for lab in np.unique(labels):
        if lab == ignore_label:
            continue
        idx = np.where(labels == lab)[0]
        if idx.size < min_cluster_size:
            continue

        n = normals[idx]
        n_unit, nrm = normalize(n, eps=1e-12)  # this normalizes the whole matrix incorrectly
        # correct per-row normalization:
        row_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(row_norm > 1e-12, n / row_norm, n)

        n_bar = n.mean(axis=0)
        n_bar, nb = normalize(n_bar)
        if nb < 1e-12:
            continue

        dots = np.clip(np.sum(n * n_bar[None, :], axis=1), -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dots))

        out.append({
            "label": int(lab),
            "n_points": int(idx.size),
            "mean_angle_deg": float(np.mean(angles_deg)),
            "median_angle_deg": float(np.median(angles_deg)),
            "p95_angle_deg": float(np.percentile(angles_deg, 95)),
        })

    out.sort(key=lambda r: r["label"])
    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Load your Nx4 (x,y,z,label) clustered data
    # points_xyzL = np.load(r"H:\CreateBIM\dataset\Output\dbscan\d2k_1_subsampled_20cm_point_clusters.npy")

    # points_xyzL = np.load(r"H:\CreateBIM\dataset\Output\dbscan\d2k_2_original_point_clusters.npy")

    points_xyzL = np.load(r"H:\CreateBIM\dataset\Output\dbscan\d2k_5_point_clusters.npy")

    # Validate planes per cluster (exclude noise=-1)
    results = validate_plane_fit(points_xyzL, ignore_label=-1, min_cluster_size=50)
    print_plane_fit_report(results)

    # Save CSV for thesis table (make sure folder exists)
    # save_plane_fit_csv(results, r"H:\CreateBIM\dataset\Output\dbscan\plane_fit_report_d2k1.csv")

    # save_plane_fit_csv(results, r"H:\CreateBIM\dataset\Output\dbscan\plane_fit_report_d2k.csv")

    save_plane_fit_csv(results, r"H:\CreateBIM\dataset\Output\dbscan\plane_fit_report_d2k3.csv")
