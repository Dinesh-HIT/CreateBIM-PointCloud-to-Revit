import json
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from typing import Tuple, List, Dict, Any, Optional

# ============================================================
# 0) Basic IO + Clustering
# ============================================================
def load_points_ply(ply_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        raise ValueError(f"PLY contains no points: {ply_path}")
    return pts


def cluster_points(points: np.ndarray, eps: float, min_samples: int = 50) -> Tuple[np.ndarray, List[int]]:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(points)
    cluster_ids = [int(cid) for cid in np.unique(labels) if cid != -1]
    return labels, cluster_ids


# ============================================================
# 1) Geometry helpers (plane basis, projection, PIP)
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v.copy() if n < eps else (v / n)


def make_plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normalize(n)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    ref = np.array([1.0, 0.0, 0.0], dtype=float) if abs(np.dot(n, up)) > 0.95 else up
    u = normalize(np.cross(ref, n))
    v = normalize(np.cross(n, u))
    return u, v


def project_to_plane_2d(x: np.ndarray, p0: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    d = x - p0
    return np.array([np.dot(d, u), np.dot(d, v)], dtype=float)


def point_in_polygon_2d(pt: np.ndarray, poly: np.ndarray) -> bool:
    x, y = pt
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def point_plane_distance(x: np.ndarray, p0: np.ndarray, n: np.ndarray) -> float:
    n = normalize(n)
    return float(abs(np.dot(n, x - p0)))


def compute_2d_obb(points2d: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    mu = points2d.mean(axis=0)
    d = np.linalg.norm(points2d - mu, axis=1)
    keep = d <= np.percentile(d, 98)
    pts = points2d[keep] if np.any(keep) else points2d

    mu = pts.mean(axis=0)
    X = pts - mu
    C = np.cov(X.T)

    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]

    Y = X @ R
    min_xy = Y.min(axis=0)
    max_xy = Y.max(axis=0)

    wh = max_xy - min_xy
    center_local = (min_xy + max_xy) / 2.0
    center2d = mu + center_local @ R.T

    width, height = float(wh[0]), float(wh[1])
    angle = float(np.arctan2(R[1, 0], R[0, 0]))
    return center2d, width, height, angle


# ============================================================
# 2) Load envelope planes (A) full planes for output, (B) wall-like for hosting
# ============================================================
def _get_corners(pl: Dict[str, Any]):
    """
    Read polygon corners under either key name used across versions.
    The pipeline writes 'corners_ccw_3d'; older files used 'corners'.
    """
    if "corners_ccw_3d" in pl:
        return pl["corners_ccw_3d"]
    if "corners" in pl:
        return pl["corners"]
    return None


def load_all_planes_raw(json_path: str) -> List[Dict[str, Any]]:
    """
    Load ALL planes from pyrevit_planes.json and keep the same schema:
    {label, normal, corners}. This is what you want to embed in the unified JSON.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    planes_out = []
    for pl in data.get("planes", []):
        corners = _get_corners(pl)
        if (corners is None) or ("normal" not in pl):
            continue
        planes_out.append({
            "label": int(pl.get("label", pl.get("id", -1))),
            "normal": pl["normal"],
            "corners": corners
        })

    if not planes_out:
        raise ValueError("No planes found in envelope JSON.")
    return planes_out


def load_wall_like_planes_for_hosting(json_path: str, wall_nz_threshold: float = 0.35) -> List[Dict[str, Any]]:
    """
    Reads pyrevit_planes.json and keeps wall-like planes: abs(nz) < wall_nz_threshold.
    Precomputes 2D polygon coords for PIP test.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    planes_out = []
    for pl in data.get("planes", []):
        label = int(pl.get("label", pl.get("id", -1)))
        n = np.array(pl["normal"], dtype=float)
        corners_raw = _get_corners(pl)
        if corners_raw is None:
            continue
        corners = np.array(corners_raw, dtype=float)
        if corners.shape[0] < 3:
            continue
        if abs(n[2]) >= wall_nz_threshold:
            continue

        p0 = corners[0]
        u, v = make_plane_basis(n)
        poly2d = np.vstack([project_to_plane_2d(c, p0, u, v) for c in corners])

        planes_out.append({
            "label": label,
            "n": n,
            "p0": p0,
            "u": u,
            "v": v,
            "corners3d": corners,
            "poly2d": poly2d
        })

    if not planes_out:
        raise ValueError("No wall-like planes kept. Increase wall_nz_threshold or check normals.")
    return planes_out


# ============================================================
# 3) Host wall assignment + opening record
# ============================================================
def find_host_plane_for_cluster(
    cluster_pts: np.ndarray,
    planes: List[Dict[str, Any]],
    tol_plane: float = 0.07
) -> Tuple[Optional[Dict[str, Any]], np.ndarray, Optional[float]]:
    c = cluster_pts.mean(axis=0)
    best_plane = None
    best_dist = None

    for pl in planes:
        dist = point_plane_distance(c, pl["p0"], pl["n"])
        if dist > tol_plane:
            continue
        c2d = project_to_plane_2d(c, pl["p0"], pl["u"], pl["v"])
        if not point_in_polygon_2d(c2d, pl["poly2d"]):
            continue
        if (best_dist is None) or (dist < best_dist):
            best_plane = pl
            best_dist = dist

    return best_plane, c, best_dist


def build_opening_record(cluster_pts: np.ndarray, host_plane: Dict[str, Any], centroid: np.ndarray) -> Dict[str, Any]:
    p0, u, v = host_plane["p0"], host_plane["u"], host_plane["v"]
    pts2d = np.vstack([project_to_plane_2d(p, p0, u, v) for p in cluster_pts])
    center2d, width, height, angle = compute_2d_obb(pts2d)
    sill_v_plane = float(np.min(pts2d[:, 1]))

    return {
        "center": centroid.tolist(),
        "width": float(width),
        "height": float(height),
        "angle_rad": float(angle),
        "sill_v_plane": sill_v_plane,
        "units": "m",

        "host_plane_label": int(host_plane["label"]),
        "host_plane_normal": host_plane["n"].tolist(),
        "host_plane_u": host_plane["u"].tolist(),
        "host_plane_v": host_plane["v"].tolist(),
        "host_plane_p0": host_plane["p0"].tolist(),
    }


def process_openings(
    points: np.ndarray,
    planes: List[Dict[str, Any]],
    opening_type: str,
    eps: float,
    min_samples: int,
    tol_plane: float
) -> List[Dict[str, Any]]:
    labels, cluster_ids = cluster_points(points, eps=eps, min_samples=min_samples)

    print(f"[INFO] {opening_type.upper()} points: {points.shape[0]}")
    print(f"[INFO] {opening_type.upper()} DBSCAN eps={eps}, min_samples={min_samples}")
    print(f"[INFO] {opening_type.upper()} clusters: {len(cluster_ids)} | Noise points: {int(np.sum(labels == -1))}")

    openings = []
    for idx, cid in enumerate(cluster_ids):
        cluster_pts = points[labels == cid]

        host_plane, centroid, dist = find_host_plane_for_cluster(
            cluster_pts, planes, tol_plane=tol_plane
        )

        if host_plane is None:
            # Keep keys consistent: center (not centroid)
            openings.append({
                "id": f"{opening_type}_{idx:03d}",
                "opening_type": opening_type,
                "cluster_id": int(cid),
                "host_plane_label": None,
                "status": "unassigned",
                "center": centroid.tolist(),
                "note": "No wall plane satisfied distance+inside checks; increase tol_plane or verify alignment."
            })
            continue

        rec = build_opening_record(cluster_pts, host_plane, centroid)
        openings.append({
            "id": f"{opening_type}_{idx:03d}",
            "opening_type": opening_type,
            "cluster_id": int(cid),
            "plane_distance_m": float(dist),
            "status": "assigned",
            **rec
        })

    return openings

# ============================================================
# 4) Main
# ============================================================
def main():
    #todo:dataset1

    # planes_json = r"G:\CreateBIM\dataset\dataset1\polygon_results\pyrevit_planes.json"
    # windows_ply = r"G:\CreateBIM\dataset\dataset1\d2k_1_original_window_transfer.ply"
    # doors_ply   = r"G:\CreateBIM\dataset\dataset1\d2k_1_original_door_transfer.ply"
    # out_path = r"G:\CreateBIM\dataset\dataset1\polygon_results\bim_input_unified.json"

    # #todo:dataset2
    # planes_json = r"G:\CreateBIM\dataset\dataset2\polygon_results\pyrevit_planes.json"
    # windows_ply = r"G:\CreateBIM\dataset\dataset2\d2k_2_window_transfer.ply"
    # out_path = r"G:\CreateBIM\dataset\dataset2\polygon_results\bim_input_unified.json" # ✅ unified output JSON (contains planes + windows + doors)

    # #todo:dataset3
    # planes_json = r"G:\CreateBIM\dataset\dataset3\polygon_results\pyrevit_planes.json"
    # windows_ply = r"G:\CreateBIM\dataset\dataset3\d2k_3_window_transfer.ply"
    # out_path = r"G:\CreateBIM\dataset\dataset3\polygon_results\bim_input_unified.json" # ✅ unified output JSON (contains planes + windows + doors)
    #
    # #todo:dataset4
    planes_json = r"G:\CreateBIM\dataset\dataset4\polygon_results\pyrevit_planes.json"
    windows_ply = r"G:\CreateBIM\dataset\dataset4\experiment_1_window_transfer.ply"
    out_path = r"G:\CreateBIM\dataset\dataset4\polygon_results\bim_input_unified.json" # ✅ unified output JSON (contains planes + windows + doors)
    #
    # #todo:dataset5
    # planes_json = r"G:\CreateBIM\dataset\dataset5\polygon_results\pyrevit_planes.json"
    # windows_ply = r"G:\CreateBIM\dataset\dataset5\d2k_5_original_window_transfer.ply"
    # out_path = r"G:\CreateBIM\dataset\dataset5\polygon_results\bim_input_unified.json"  # ✅ unified output JSON (contains planes + windows + doors)
    #

    #TODO: DBSCAN parameters
    window_eps = 0.50
    window_min_samples = 10

    door_eps = 0.30
    door_min_samples = 25

    #todo host assignment controls
    wall_nz_threshold = 0.5    #dataset 1 and 3 = 0.35
    tol_plane = 0.12  #0.07 to 0.10 or 0.12 #dataset 1 and 3 = 0.1

    # (A) Load ALL planes for unified output
    planes_all = load_all_planes_raw(planes_json)

    # (B) Load only wall-like planes for hosting openings
    planes_walls = load_wall_like_planes_for_hosting(planes_json, wall_nz_threshold=wall_nz_threshold)

    # process windows
    window_pts = load_points_ply(windows_ply)
    windows = process_openings(
        points=window_pts,
        planes=planes_walls,
        opening_type="window",
        eps=window_eps,
        min_samples=window_min_samples,
        tol_plane=tol_plane
    )
#TODO : doors
    # process doors
    # door_pts = load_points_ply(doors_ply)
    # doors = process_openings(
    #     points=door_pts,
    #     planes=planes_walls,
    #     opening_type="door",
    #     eps=door_eps,
    #     min_samples=door_min_samples,
    #     tol_plane=tol_plane
    # )

    out = {
        "meta": {
            "source_envelope_json": planes_json,
            "source_windows_ply": windows_ply,
#            "source_doors_ply": doors_ply,

            "window_dbscan_eps_m": float(window_eps),
            "window_dbscan_min_samples": int(window_min_samples),

            "door_dbscan_eps_m": float(door_eps),
            "door_dbscan_min_samples": int(door_min_samples),

            "tol_plane_m": float(tol_plane),
            "wall_nz_threshold": float(wall_nz_threshold),

            "units": "m"
        },

        # ✅ envelope planes are embedded here
        "planes": planes_all,

        # ✅ openings are embedded here
        "windows": windows,
#        "doors": doors
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    assigned_windows = sum(1 for o in windows if o.get("host_plane_label") is not None)
#    assigned_doors = sum(1 for o in doors if o.get("host_plane_label") is not None)

    print(f"[RESULT] Windows assigned: {assigned_windows}/{len(windows)}")
#   print(f"[RESULT] Doors assigned: {assigned_doors}/{len(doors)}")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()