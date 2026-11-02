import json
import numpy as np
from matplotlib.path import Path

# ============================================================
# 1) Load BIM polygons exported from polygon_construction.py
# ============================================================

def load_bim_polygons(pyrevit_json_path):
    """
    Loads polygon faces exported in PyRevit JSON format.
    Returns: dict[label] -> {normal, corners}
    """
    with open(pyrevit_json_path, "r") as f:
        data = json.load(f)

    polygons = {}
    for p in data["planes"]:
        lbl = int(p["label"])
        polygons[lbl] = {
            "normal": np.asarray(p["normal"], dtype=float),
            "corners": np.asarray(p["corners"], dtype=float)
        }
    return polygons


# ============================================================
# 2) Geometry helpers (for UV projection)
# ============================================================

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n

def build_plane_basis(normal):
    """
    Build stable orthonormal (u,v) basis for plane with normal n.
    """
    n = normalize(normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(n, ref))
    v = np.cross(n, u)
    return u, v

def project_to_plane_uv(points, normal, origin=None):
    """
    Project 3D points to 2D (u,v) coordinates on the plane.
    """
    u, v = build_plane_basis(normal)
    if origin is None:
        origin = points.mean(axis=0)
    X = points - origin
    return np.column_stack((X @ u, X @ v))


# ============================================================
# 3) Important validation metrics
# ============================================================

def polygon_is_closed_bim(corners):
    """
    BIM polygons are implicitly closed: last vertex connects to first.
    Closure is satisfied if polygon has at least 3 vertices.
    """
    return corners is not None and corners.shape[0] >= 3

def polygon_planarity_rms(corners, normal):
    """
    RMS vertex-to-plane deviation (geometric consistency).
    """
    if corners is None or len(corners) < 3:
        return np.inf
    n = normalize(normal)
    centroid = corners.mean(axis=0)
    dists = np.abs((corners - centroid) @ n)
    return float(np.sqrt(np.mean(dists ** 2)))

def segments_intersect_2d(a, b, c, d, eps=1e-12):
    """
    Proper 2D segment intersection test for segments ab and cd (general case).
    """
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    return (o1 * o2 < -eps) and (o3 * o4 < -eps)

def has_self_intersection(poly_uv):
    """
    Checks if polygon has self intersections in 2D.
    """
    n = len(poly_uv)
    if n < 4:
        return False

    for i in range(n):
        a = poly_uv[i]
        b = poly_uv[(i + 1) % n]

        for j in range(i + 1, n):
            # Skip adjacent edges
            if abs(i - j) <= 1:
                continue
            # Skip first-last adjacency
            if i == 0 and j == n - 1:
                continue

            c = poly_uv[j]
            d = poly_uv[(j + 1) % n]

            if segments_intersect_2d(a, b, c, d):
                return True
    return False

def face_coverage_percent(corners, normal, pts, tol=0.02):
    """
    Percentage of plane-labeled points inside polygon (after projection).
    """
    if pts is None or pts.shape[0] < 10 or corners is None or len(corners) < 3:
        return 0.0

    origin = corners.mean(axis=0)
    poly_uv = project_to_plane_uv(corners, normal, origin=origin)
    pts_uv  = project_to_plane_uv(pts, normal, origin=origin)

    poly_path = Path(poly_uv)
    inside = poly_path.contains_points(pts_uv, radius=tol)

    return 100.0 * float(inside.sum()) / float(pts.shape[0])


# ============================================================
# 4) Main validation routine (minimal)
# ============================================================

def validate_bim_polygons_minimal(pyrevit_json_path, pts_all, labels_all, tol_inside=0.02):
    """
    Minimal validation:
      1) Polygon validity (implicit closure + no self-intersection)
      2) Planarity RMS error
      3) Face coverage percent
    """
    polygons = load_bim_polygons(pyrevit_json_path)
    results = []

    for lbl, poly in polygons.items():
        corners = poly["corners"]
        normal  = poly["normal"]
        pts     = pts_all[labels_all == lbl]

        # Validity checks
        closed_bim = polygon_is_closed_bim(corners)

        origin = corners.mean(axis=0) if corners is not None and len(corners) > 0 else None
        poly_uv = project_to_plane_uv(corners, normal, origin=origin) if origin is not None else None

        self_int = has_self_intersection(poly_uv) if poly_uv is not None else True
        valid_polygon = closed_bim and (not self_int)

        # Geometric + data agreement metrics
        planarity_rms = polygon_planarity_rms(corners, normal)
        coverage = face_coverage_percent(corners, normal, pts, tol=tol_inside)

        results.append({
            "label": lbl,
            "n_pts_cluster": int(pts.shape[0]),
            "n_vertices": int(len(corners)),
            "valid_polygon": bool(valid_polygon),
            "self_intersection": bool(self_int),
            "planarity_rms_m": float(planarity_rms),
           # "coverage_percent": float(coverage),
        })

    return results


# ============================================================
# 5) Example usage
# ============================================================

if __name__ == "__main__":
    PYREVIT_JSON = r"H:\CreateBIM\dataset\Output\dbscan\density_map\d2k_1_output\pyrevit_planes.json"
    POINTS_XYZL  = r"H:\CreateBIM\dataset\Output\dbscan\d2k_1_original_point_clusters.npy"

    # PYREVIT_JSON = r"H:\CreateBIM\dataset\Output\dbscan\density_map\d2k_2_output\pyrevit_planes.json"
    # POINTS_XYZL = r"H:\CreateBIM\dataset\Output\dbscan\d2k_2_original_point_clusters.npy"

    # PYREVIT_JSON = r"H:\CreateBIM\dataset\Output\dbscan\density_map\d2k_5_output\pyrevit_planes.json"
    # POINTS_XYZL  = r"H:\CreateBIM\dataset\Output\dbscan\d2k_5_point_clusters.npy"

    points_xyzL = np.load(POINTS_XYZL)
    pts_all = points_xyzL[:, :3].astype(float)
    labels_all = points_xyzL[:, 3].astype(int)

    results = validate_bim_polygons_minimal(
        PYREVIT_JSON,
        pts_all,
        labels_all,
        tol_inside=0.02
    )

    print("\n[BIM Polygon Validation Results - Minimal]")
    print("label | pts | verts | valid | selfInt | planarity(m) | coverage(%)")
    print("-" * 78)
    for r in results:
        print(f"{r['label']:5d} | "
              f"{r['n_pts_cluster']:5d} | "
              f"{r['n_vertices']:5d} | "
              f"{str(r['valid_polygon']):5s} | "
              f"{str(r['self_intersection']):7s} | "
              f"{r['planarity_rms_m']:.4e} | ")
              #f"{r['coverage_percent']:7.1f}")
