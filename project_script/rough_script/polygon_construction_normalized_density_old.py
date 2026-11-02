import os
import json
import math
from itertools import combinations
from collections import defaultdict
import numpy as np
import pyvista as pv

# -----------------------------------------------------------------------------------------
# 0) CONFIG (edit paths only)
# -----------------------------------------------------------------------------------------

# NPY_PATH = r"G:\CreateBIM\dataset\dataset1\dbscan_results\d2k_1_original_point_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset1\dbscan_results\d2k_1_original_noisy_points_transfer.npy"

# NPY_PATH = r"G:\CreateBIM\dataset\dataset2\dbscan_results\d2k_2_point_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset2\dbscan_results\d2k_2_noisy_points_transfer.npy"

# NPY_PATH = r"G:\CreateBIM\dataset\dataset3\dbscan_results\d2k_3_point_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset3\dbscan_results\d2k_3_noisy_points_transfer.npy"

# NPY_PATH  = r"G:\CreateBIM\dataset\dataset4\dbscan_results\experiment_1_voxel_points_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset4\dbscan_results\experiment_1_voxel_noisy_points_transfer.npy"

NPY_PATH  = r"G:\CreateBIM\dataset\dataset5\dbscan_results\experiment_2_points_clusters_transfer.npy"
NPY_NOISY = r"G:\CreateBIM\dataset\dataset5\dbscan_results\experiment_2_noisy_points_transfer.npy"

# Outputs
OUT_DIR = r"G:\CreateBIM\dataset\dataset5\polygon_results"
OUT_CSV = os.path.join(OUT_DIR, "segments_density.csv")
OUT_JSON = os.path.join(OUT_DIR, "geometry_output.json")
OUT_PNG = os.path.join(OUT_DIR, "view_density_map_after_filter.png")

# Prefilter (BEFORE filtering): ONLY the density-map PNG
SAVE_PREFILTER_PNG = True
OUT_PNG_ALL = os.path.join(OUT_DIR, "view_density_map_all.png")

# Postfilter (AFTER filtering): keep usual outputs
SAVE_POSTFILTER_CSV = True
SAVE_POSTFILTER_JSON = True
SAVE_POSTFILTER_PNG = True

# PyRevit export (AFTER filtering): polygon corners per plane (not forced to rectangle)
OUT_PYREVIT_JSON = os.path.join(OUT_DIR, "pyrevit_planes.json")
SAVE_PYREVIT_JSON = True

# ---------------- Polygon visualization (AFTER polygon formation) ----------------
OUT_PNG_POLY = os.path.join(OUT_DIR, "view_snapshot_polygons.png")
SAVE_POLYGON_PNG = True

POLY_DRAW_SELECTED_SEGS = True    # draw selected segments under polygons
POLY_DRAW_ALL_SEGS_FAINT = False  # draw non-selected too (can clutter)

POLY_EDGE_WIDTH = 8.0
POLY_EDGE_OPACITY = 1.0
POLY_FILL_ENABLE = True
POLY_FILL_OPACITY = 0.18
POLY_OFFSET_ALONG_NORMAL = 0.002  # meters, avoids z-fighting

# Plane polygonization from FILTERED segments
POLY_SNAP_EPS_UV = 0.05          # snapping in plane (meters): merges nearly-identical vertices
POLY_MIN_LOOP_AREA_UV = 0.20     # min |area| in plane-UV units (m^2) to keep a loop
POLY_MIN_EDGE_LEN_UV = 0.03      # drop tiny edges after snapping (m)
POLY_COLLINEAR_EPS = 1e-3        # collinearity tolerance for vertex simplification (radians-ish)
POLY_MAX_VERTS = 2000            # safety cap per loop

# Geometry / robustness
BBOX_PAD = 0.50
BBOX_USE_NOISY_FOR_BOUNDS = True
USE_ROBUST_BBOX_BOUNDS = True
BBOX_ROBUST_PCT = (0.5, 99.5)
PLANE_DIST_EPS = 0.05
DET_EPS = 1e-10
MIN_SEG_LEN = 1e-3

# Vertex (corner) estimation
VERTEX_REFINE_ENABLE = False     # set True to enable m>=3 LLS corner refinement
CORNER_CLUSTER_RADIUS = 0.10     # meters (only used if VERTEX_REFINE_ENABLE=True)
VERTEX_LLS_RESIDUAL_THRESH = None  # e.g. 1e-5 to reject high-residual intersections

# Noisy-to-segment voting
MAX_SNAP_DIST = 0.15
USE_DISTANCE_WEIGHT = False
SIGMA_DIST = 0.20

# ------------------ Option A: Segment filtering ------------------
FILTER_MODE = "threshold"  # "top_pct" or "threshold"

# Use normalized density for filtering? (recommended)
# True  => filter uses density_norm in [0,1]
# False => filter uses raw density = votes / length
USE_NORMALIZED_DENSITY_FOR_FILTER = True

TOP_PCT = 65

# Threshold filtering:
# If USE_NORMALIZED_DENSITY_FOR_FILTER=True  => use DENSITY_NORM_THRESH (0..1)
# If USE_NORMALIZED_DENSITY_FOR_FILTER=False => use DENSITY_THRESH (raw)
DENSITY_THRESH = 2.0        # raw density threshold (votes/m)
DENSITY_NORM_THRESH = 0.10 # normalized threshold (0..1)

# Visualization
PLANE_OPACITY = 0.08
NOISY_OPACITY = 0.25
NOISY_POINT_SIZE = 2.0
NONSELECT_OPACITY = 0.15
NONSELECT_LINE_W = 2.0
SELECT_LINE_W_MIN = 4.0
SELECT_LINE_W_MAX = 10.0
DENSITY_VIS_MODE = "linear"  # "linear", "log1p", or "rank"
DENSITY_CLIP_PCT = (5, 95)
HEATMAP_CMAP = "turbo"

# Prefilter visualization controls
DENSITY_CLIP_PCT_ALL = (10, 95)
BBOX_VIS_MARGIN_RATIO = 0.05
PREFILTER_SANITY_FILTER = True

# Density stabilization
LENGTH_EPS_FOR_DENSITY = 0.05

# Rendering
OFFSCREEN_RENDER = True

# Camera control (prefilter PNG)
PREFILTER_CAMERA_ZOOM = 1.0
PREFILTER_CAMERA_ISO = True


# -----------------------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------------------
def info(msg): print("[INFO]", msg)
def warn(msg): print("[WARN]", msg)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_points_clusters(npy_path):
    data = np.load(npy_path)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("Expected (N,4+) array for clusters: x,y,z,label")
    pts = data[:, :3].astype(float)
    labels = data[:, 3].astype(int)
    return pts, labels

def load_noisy_points(npy_path):
    data = np.load(npy_path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Expected (M,3+) array for noisy points.")
    return data[:, :3].astype(float)

def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v, 0.0
    return v / n, n

def plane_fit_pca_tls(points):
    """Total least squares plane via PCA. Returns (centroid, unit_normal, d) for n·x + d = 0."""
    c = points.mean(axis=0)
    X = points - c
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1, :]
    n, _ = normalize(n)
    d = -float(np.dot(n, c))
    return c, n, d

def triple_plane_vertex(n1, d1, n2, d2, n3, d3):
    """
    Solve [n1^T; n2^T; n3^T] v = -[d1,d2,d3]^T.
    Returns (v, ok).
    """
    A = np.vstack([n1, n2, n3])
    det = float(np.linalg.det(A))
    if abs(det) < DET_EPS:
        return None, False
    b = -np.array([d1, d2, d3], dtype=float)
    v = np.linalg.solve(A, b)
    return v, True


# -----------------------------------------------------------------------------------------
# 2.x) LLS vertex utilities (triplet seeds + optional m>=3 refinement)
# -----------------------------------------------------------------------------------------
def find_best_fit_vertex_lls(normals, offsets, residual_thresh=None, det_eps=None):
    """Least-squares plane intersection.
    Solves A x ≈ b where rows of A are plane normals and b = -offsets.
    - If det_eps is given and A is 3x3, near-singular triplets are rejected (keeps legacy behavior).
    - If residual_thresh is given, rejects solutions with large squared residual ||A x - b||^2.
    Returns x (shape (3,)) or None.
    """
    A = np.vstack(normals).astype(float)  # (m,3)
    b = -np.array(offsets, dtype=float)   # (m,)

    if det_eps is not None and A.shape[0] == 3:
        det = float(np.linalg.det(A))
        if abs(det) < det_eps:
            return None

    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None

    if x is None or not np.all(np.isfinite(x)):
        return None

    if residual_thresh is not None:
        r = A @ x - b
        if float(np.dot(r, r)) > float(residual_thresh):
            return None

    return x

def cluster_vertices_by_proximity(raw_vertices, radius):
    """Simple greedy spatial clustering for raw corner candidates (as in the m>=3 LLS pipeline)."""
    clusters = []
    radius2 = float(radius) * float(radius)

    for v in raw_vertices:
        p = v["point"]
        planes = v["planes"]
        assigned = False

        for cl in clusters:
            if float(np.sum((p - cl["center"]) ** 2)) <= radius2:
                cl["points"].append(p)
                cl["planes"].update(planes)
                cl["center"] = np.mean(cl["points"], axis=0)
                assigned = True
                break

        if not assigned:
            clusters.append({
                "points": [p],
                "planes": set(planes),
                "center": p.copy(),
            })

    return clusters

def point_plane_abs(n, d, p):
    return abs(float(np.dot(n, p) + d))

def segment_point_distance(p, a, b):
    """Distance from point p to segment ab; also returns projection parameter t in [0,1]."""
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-18:
        return float(np.linalg.norm(p - a)), 0.0
    t = float(np.dot(p - a, ab) / ab2)
    t_clamped = max(0.0, min(1.0, t))
    proj = a + t_clamped * ab
    return float(np.linalg.norm(p - proj)), t_clamped

def unique_points(points, eps=1e-6):
    """Deduplicate 3D points by grid hashing."""
    if len(points) == 0:
        return np.empty((0, 3), float)
    pts = np.asarray(points, float)
    key = np.round(pts / eps).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[np.sort(idx)]

def choose_plane_basis(n, pts3d):
    """
    Choose a stable in-plane basis (u,v) using PCA of pts3d projected onto plane.
    If degenerate, choose arbitrary.
    """
    n, _ = normalize(n)
    if pts3d.shape[0] >= 2:
        c = pts3d.mean(axis=0)
        X = pts3d - c
        X = X - (X @ n)[:, None] * n[None, :]
        if np.linalg.norm(X) > 1e-10:
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            u = vh[0, :]
            u = u - np.dot(u, n) * n
            u, un = normalize(u)
            if un > 1e-12:
                v = np.cross(n, u)
                v, _ = normalize(v)
                return u, v
    axes = np.eye(3)
    dots = np.abs(axes @ n)
    a = axes[int(np.argmin(dots))]
    u = np.cross(n, a)
    u, _ = normalize(u)
    v = np.cross(n, u)
    v, _ = normalize(v)
    return u, v

def density_transform(vals, mode="log1p"):
    vals = np.asarray(vals, float)
    if mode == "linear":
        return vals
    if mode == "log1p":
        return np.log1p(np.clip(vals, 0, None))
    if mode == "rank":
        order = np.argsort(vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.linspace(0, 1, len(vals), endpoint=True)
        return ranks
    raise ValueError("Unknown DENSITY_VIS_MODE: %s" % mode)

def segment_in_bbox(a, b, bb_min, bb_max, margin):
    """Sanity filter for visualization: keep only segments whose endpoints are within bbox ± margin."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    bb_min = np.asarray(bb_min, float)
    bb_max = np.asarray(bb_max, float)
    lo = bb_min - float(margin)
    hi = bb_max + float(margin)
    return bool(np.all(a >= lo) and np.all(a <= hi) and np.all(b >= lo) and np.all(b <= hi))


# -----------------------------------------------------------------------------------------
# Polygon visualization helpers (3D loops -> meshes)
# -----------------------------------------------------------------------------------------
def loop3d_to_closed_polyline(loop3d):
    """Create a closed PolyData polyline (edges) from ordered loop points."""
    pts = np.asarray(loop3d, float)
    if pts.shape[0] < 2:
        return None
    n = pts.shape[0]
    lines = []
    for i in range(n - 1):
        lines.append([2, i, i + 1])
    lines.append([2, n - 1, 0])
    poly = pv.PolyData(pts)
    poly.lines = np.array(lines, dtype=np.int64)
    return poly

def loop3d_to_filled_polygon(loop3d):
    """Create a filled (triangulated) polygon mesh from ordered loop points."""
    pts = np.asarray(loop3d, float)
    if pts.shape[0] < 3:
        return None
    n = pts.shape[0]
    faces = np.hstack([[n], np.arange(n, dtype=np.int64)]).astype(np.int64)
    poly = pv.PolyData(pts, faces=faces)
    try:
        poly = poly.triangulate()
    except Exception:
        pass
    return poly


# -----------------------------------------------------------------------------------------
# Polygonization for PyRevit export (build plane polygons from FILTERED intersection segments)
# -----------------------------------------------------------------------------------------
def _poly_area_uv(loop_uv):
    """Compute signed area of a 2D polygon using shoelace formula."""
    x = loop_uv[:, 0]
    y = loop_uv[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _remove_collinear_uv(loop_uv, eps=POLY_COLLINEAR_EPS, min_edge=POLY_MIN_EDGE_LEN_UV):
    """Remove collinear vertices and vertices forming very short edges."""
    uv = np.asarray(loop_uv, float)
    if uv.shape[0] < 4:
        return uv
    if np.linalg.norm(uv[0] - uv[-1]) < 1e-12:
        uv = uv[:-1]

    def angle(u, v):
        """Angle between vectors u and v in radians."""
        nu = np.linalg.norm(u)
        nv = np.linalg.norm(v)
        if nu < 1e-12 or nv < 1e-12:
            return 0.0
        c = float(np.dot(u, v) / (nu * nv))
        c = max(-1.0, min(1.0, c))
        return math.acos(c)

    changed = True
    while changed and uv.shape[0] >= 3:
        changed = False
        keep = []
        n = uv.shape[0]
        for i in range(n):
            p_prev = uv[(i - 1) % n]
            p = uv[i]
            p_next = uv[(i + 1) % n]
            v1 = p - p_prev
            v2 = p_next - p
            if np.linalg.norm(v1) < min_edge or np.linalg.norm(v2) < min_edge:
                changed = True
                continue
            ang = angle(v1, v2)
            if ang < eps or abs(ang - math.pi) < eps:
                changed = True
                continue
            keep.append(p)
        uv = np.asarray(keep, float)
        if uv.shape[0] < 3:
            break
    return uv

def ccw_angle_diff(alpha, beta):
    """Compute CCW angle difference from β to α: (α - β) mod 2π in [0, 2π)."""
    return (alpha - beta) % (2 * math.pi)

def plane_polygons_from_segments(plane_label, plane_info, segs_for_plane):
    """
    Extract polygon boundary loops from segments lying on a plane.

    Steps:
    1. Project all segment endpoints to plane-local 2D (UV) coordinates
    2. Snap nearby vertices to handle floating-point errors
    3. Build planar straight-line graph (PSLG) from snapped vertices
    4. Trace CCW boundary cycles using sorted adjacency
    5. Return the largest cycle as the main boundary (NO-OPENINGS)
    """
    n = np.asarray(plane_info["normal"], float)
    d = float(plane_info["d"])

    pts3 = []
    for a, b in segs_for_plane:
        pts3.append(np.asarray(a, float))
        pts3.append(np.asarray(b, float))
    pts3 = np.asarray(pts3, float)
    if pts3.shape[0] < 6:
        return None

    c = pts3.mean(axis=0)
    t0 = -(np.dot(n, c) + d)
    p0 = c + t0 * n  # plane origin

    uvec, vvec = choose_plane_basis(n, pts3)

    rel = pts3 - p0[None, :]
    uv_all = np.column_stack([rel @ uvec, rel @ vvec])

    # 1) Snapping
    eps = float(POLY_SNAP_EPS_UV)
    keys = np.round(uv_all / eps).astype(np.int64)

    key_to_id = {}
    node_uv_sum = []
    node_count = []

    def get_node_id(k, uv):
        k = (int(k[0]), int(k[1]))
        if k in key_to_id:
            nid = key_to_id[k]
            node_uv_sum[nid] += uv
            node_count[nid] += 1
            return nid
        nid = len(node_uv_sum)
        key_to_id[k] = nid
        node_uv_sum.append(np.array(uv, float))
        node_count.append(1)
        return nid

    node_ids = [get_node_id(keys[i], uv_all[i]) for i in range(uv_all.shape[0])]
    node_uv = np.array([node_uv_sum[i] / max(1, node_count[i]) for i in range(len(node_uv_sum))], float)

    # 2) Build undirected edges
    edges = set()
    for e in range(0, len(node_ids), 2):
        a = node_ids[e]
        b = node_ids[e + 1]
        if a == b:
            continue
        if np.linalg.norm(node_uv[b] - node_uv[a]) < float(POLY_MIN_EDGE_LEN_UV):
            continue
        edges.add((a, b) if a < b else (b, a))

    if len(edges) < 3:
        return None

    # 3) Adjacency with sorted outgoing angles
    adj = {i: [] for i in range(node_uv.shape[0])}
    for a, b in edges:
        ua, va = node_uv[a]
        ub, vb = node_uv[b]
        ang_ab = math.atan2(vb - va, ub - ua)
        ang_ba = math.atan2(va - vb, ua - ub)
        adj[a].append((b, ang_ab))
        adj[b].append((a, ang_ba))

    for k in adj:
        adj[k].sort(key=lambda t: t[1])

    visited_dir = set()
    cycles = []

    def find_next_ccw(curr_b, ang_in):
        outs = adj.get(curr_b, [])
        if not outs:
            return None
        import bisect
        angles = [ang for _, ang in outs]
        idx = bisect.bisect_right(angles, ang_in)
        nb, ang_out = outs[idx % len(outs)]
        return nb, ang_out

    # 4) Trace cycles
    for a in adj:
        for b, ang_ab in adj[a]:
            start = (a, b)
            if start in visited_dir:
                continue

            loop = []
            curr_a, curr_b = a, b

            for _ in range(POLY_MAX_VERTS):
                visited_dir.add((curr_a, curr_b))
                loop.append(curr_a)

                ua, va = node_uv[curr_a]
                ub, vb = node_uv[curr_b]
                ang_in = math.atan2(va - vb, ua - ub)

                nxt = find_next_ccw(curr_b, ang_in)
                if nxt is None:
                    break
                nxt_c, _ = nxt
                curr_a, curr_b = curr_b, nxt_c

                if (curr_a, curr_b) == start:
                    loop.append(curr_a)
                    break

            uniq_ids = list(dict.fromkeys(loop))
            if len(uniq_ids) < 3:
                continue

            uv_loop = np.array([node_uv[i] for i in loop], float)
            if np.linalg.norm(uv_loop[0] - uv_loop[-1]) < 1e-12:
                uv_loop = uv_loop[:-1]
            if uv_loop.shape[0] < 3:
                continue

            uv_loop = _remove_collinear_uv(uv_loop)
            if uv_loop.shape[0] < 3:
                continue

            area = _poly_area_uv(uv_loop)
            if abs(area) < float(POLY_MIN_LOOP_AREA_UV):
                continue

            coords = [tuple(np.round(p, 6)) for p in uv_loop]
            best = None
            for seq in (coords, list(reversed(coords))):
                min_i = min(range(len(seq)), key=lambda i: seq[i])
                rot = tuple(seq[min_i:] + seq[:min_i])
                if (best is None) or (rot < best):
                    best = rot
            cycles.append((best, uv_loop, area))

    if not cycles:
        return None

    # 6) Keep only the largest loop for each key
    by_key = {}
    for key, uv_loop, area in cycles:
        if key not in by_key or abs(area) > abs(by_key[key][1]):
            by_key[key] = (uv_loop, area)

    loops_uv = [(uv, area) for uv, area in by_key.values()]
    loops_uv.sort(key=lambda t: -abs(t[1]))

    outer_uv, outer_area = loops_uv[0]

    # 7) Back-project to 3D
    outer_3d = p0 + outer_uv[:, 0:1] * uvec + outer_uv[:, 1:2] * vvec

    meta = {
        "origin": p0.tolist(),
        "u": uvec.tolist(),
        "v": vvec.tolist(),
        "uv_snap_eps": float(POLY_SNAP_EPS_UV)
    }
    return {
        "basis": meta,
        "loops_uv": [outer_uv],
        "loops_3d": [outer_3d],
        "loop_areas": [outer_area]
    }

def round_vec3(pt, nd=3):
    """Round 3D point coordinates for clean JSON output."""
    return [round(float(pt[0]), nd), round(float(pt[1]), nd), round(float(pt[2]), nd)]


# -----------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------
def main():
    ensure_dir(OUT_DIR)

    info(f"Loading clusters: {NPY_PATH}")
    pts_all, labels_all = load_points_clusters(NPY_PATH)
    info(f"Loading noisy:    {NPY_NOISY}")
    noisy_pts = load_noisy_points(NPY_NOISY)

    uniq_labels = np.unique(labels_all)
    info(f"Unique clusters (planes): {len(uniq_labels)}")

    bb_pts = pts_all
    if BBOX_USE_NOISY_FOR_BOUNDS and (noisy_pts is not None) and (noisy_pts.shape[0] > 0):
        bb_pts = np.vstack([pts_all, noisy_pts])

    if USE_ROBUST_BBOX_BOUNDS:
        lo_p, hi_p = BBOX_ROBUST_PCT
        mins = np.percentile(bb_pts, float(lo_p), axis=0)
        maxs = np.percentile(bb_pts, float(hi_p), axis=0)
    else:
        mins = bb_pts.min(axis=0)
        maxs = bb_pts.max(axis=0)

    bbox_min = mins - BBOX_PAD
    bbox_max = maxs + BBOX_PAD

    # 1) Fit planes
    plane_info = {}
    for lab in uniq_labels:
        pts = pts_all[labels_all == lab]
        if pts.shape[0] < 10:
            warn(f"Cluster {lab} has too few points ({pts.shape[0]}), skipping.")
            continue
        c, n, d = plane_fit_pca_tls(pts)
        plane_info[int(lab)] = {"centroid": c, "normal": n, "d": d, "num_pts": int(pts.shape[0])}
    plane_labels = sorted(plane_info.keys())
    info(f"Fitted planes: {len(plane_labels)}")

    # 2) Corner vertices from plane intersections
    raw_vertices = []
    for i, j, k in combinations(plane_labels, 3):
        normals = [plane_info[i]["normal"], plane_info[j]["normal"], plane_info[k]["normal"]]
        offsets = [plane_info[i]["d"], plane_info[j]["d"], plane_info[k]["d"]]

        v = find_best_fit_vertex_lls(
            normals, offsets,
            residual_thresh=VERTEX_LLS_RESIDUAL_THRESH,
            det_eps=DET_EPS,
        )
        if v is None:
            continue

        if np.any(v < bbox_min) or np.any(v > bbox_max):
            continue
        if (point_plane_abs(normals[0], offsets[0], v) > PLANE_DIST_EPS or
                point_plane_abs(normals[1], offsets[1], v) > PLANE_DIST_EPS or
                point_plane_abs(normals[2], offsets[2], v) > PLANE_DIST_EPS):
            continue

        raw_vertices.append({"point": v, "planes": {i, j, k}})

    if not VERTEX_REFINE_ENABLE:
        vertices = [(rv["point"], tuple(sorted(rv["planes"]))) for rv in raw_vertices]
        info(f"Candidate vertices (triple-plane, filtered): {len(vertices)}")
    else:
        corner_clusters = cluster_vertices_by_proximity(raw_vertices, radius=CORNER_CLUSTER_RADIUS)
        refined = []
        for cl in corner_clusters:
            if len(cl["planes"]) < 3:
                continue

            normals = [plane_info[lbl]["normal"] for lbl in cl["planes"]]
            offsets = [plane_info[lbl]["d"] for lbl in cl["planes"]]

            v_ref = find_best_fit_vertex_lls(
                normals, offsets,
                residual_thresh=VERTEX_LLS_RESIDUAL_THRESH,
                det_eps=None,
            )
            if v_ref is None:
                continue
            if np.any(v_ref < bbox_min) or np.any(v_ref > bbox_max):
                continue

            ok = True
            for n, d in zip(normals, offsets):
                if point_plane_abs(n, d, v_ref) > PLANE_DIST_EPS:
                    ok = False
                    break
            if not ok:
                continue

            refined.append((v_ref, tuple(sorted(cl["planes"]))))

        vertices = refined
        info(f"Candidate vertices (m>=3 refined, filtered): {len(vertices)}")

    pair_to_pts = defaultdict(list)
    for v, planes in vertices:
        for a, b in combinations(list(planes), 2):
            pair_to_pts[tuple(sorted((a, b)))].append(v)

    # 3) Build segments
    segments = []
    for i, j in combinations(plane_labels, 2):
        pair = tuple(sorted((i, j)))
        pts = pair_to_pts.get(pair, [])
        if len(pts) < 2:
            continue

        pts_u = unique_points(pts, eps=1e-5)
        if pts_u.shape[0] < 2:
            continue

        ni = plane_info[i]["normal"]
        nj = plane_info[j]["normal"]
        dirv = np.cross(ni, nj)
        dirv, dnorm = normalize(dirv)
        if dnorm < 1e-10:
            continue

        tvals = pts_u @ dirv
        order = np.argsort(tvals)
        pts_sorted = pts_u[order]
        for a, b in zip(pts_sorted[:-1], pts_sorted[1:]):
            L = float(np.linalg.norm(b - a))
            if L < MIN_SEG_LEN:
                continue
            segments.append({"pair": pair, "a": a, "b": b, "length": L})
    info(f"Segments built (from vertices): {len(segments)}")

    if len(segments) == 0:
        warn("No segments were built. Check PLANE_DIST_EPS / clustering / vertex generation.")
        return

    #todo: density calculation 4) Vote by projecting noisy points to nearest segment (snap gate)
    counts = np.zeros(len(segments), dtype=float)
    for p in noisy_pts:
        best_idx = -1
        best_dist = 1e9
        for si, s in enumerate(segments):
            d, _ = segment_point_distance(p, s["a"], s["b"])
            if d < best_dist:
                best_dist = d
                best_idx = si
        if best_idx < 0:
            continue
        if best_dist > MAX_SNAP_DIST:
            continue

        if USE_DISTANCE_WEIGHT:
            w = math.exp(-(best_dist ** 2) / (2.0 * (SIGMA_DIST ** 2)))
            counts[best_idx] += w
        else:
            counts[best_idx] += 1.0

    lengths = np.array([s["length"] for s in segments], dtype=float)

    # Raw density = votes per unit length
    density = counts / np.maximum(lengths, float(LENGTH_EPS_FOR_DENSITY))

    # --- NEW: normalized density in [0,1] ---
    max_density = float(np.max(density)) if density.size > 0 else 0.0
    if max_density > 1e-12:
        density_norm = density / max_density
    else:
        density_norm = np.zeros_like(density, dtype=float)

    # Choose which density drives filtering + visualization
    density_for_filter = density_norm if USE_NORMALIZED_DENSITY_FOR_FILTER else density

    # 5a) PREFILTER PNG ONLY (ALL segments) -----------------------------------------
    if SAVE_PREFILTER_PNG:
        try:
            dens_vis_all = density_transform(density_for_filter, DENSITY_VIS_MODE)

            if DENSITY_CLIP_PCT_ALL is not None and len(dens_vis_all) >= 3:
                lo, hi = np.percentile(dens_vis_all, DENSITY_CLIP_PCT_ALL)
                lo, hi = float(lo), float(hi)
                if hi <= lo + 1e-12:
                    lo, hi = float(dens_vis_all.min()), float(dens_vis_all.max() + 1e-9)
            else:
                lo, hi = float(dens_vis_all.min()), float(dens_vis_all.max() + 1e-9)

            bb_pts2 = pts_all
            if noisy_pts.shape[0] > 0:
                bb_pts2 = np.vstack([bb_pts2, noisy_pts])
            bb_min = np.percentile(bb_pts2, 1, axis=0)
            bb_max = np.percentile(bb_pts2, 99, axis=0)
            bbox_diag = float(np.linalg.norm(bb_max - bb_min))
            margin = bbox_diag * float(BBOX_VIS_MARGIN_RATIO)

            seg_ids_all = list(range(len(segments)))
            if PREFILTER_SANITY_FILTER:
                seg_ids_vis = [si for si in seg_ids_all
                               if segment_in_bbox(segments[si]["a"], segments[si]["b"], bb_min, bb_max, margin)]
            else:
                seg_ids_vis = seg_ids_all

            plot_all = pv.Plotter(off_screen=OFFSCREEN_RENDER, window_size=(1400, 900))

            center_vis = 0.5 * (bb_min + bb_max)
            size = bbox_diag * 0.9 if bbox_diag > 1e-9 else 10.0

            #todo:plane show

            # for lab in plane_labels:
            #     n = plane_info[lab]["normal"]
            #     d = plane_info[lab]["d"]
            #     t0 = -(np.dot(n, center_vis) + d)
            #     p0 = center_vis + t0 * n
            #     plane = pv.Plane(center=p0, direction=n, i_size=size, j_size=size)
            #     plot_all.add_mesh(plane, color="tan", opacity=PLANE_OPACITY)

            if noisy_pts.shape[0] > 0:
                cloud = pv.PolyData(noisy_pts)
                plot_all.add_mesh(cloud, color="red", opacity=NOISY_OPACITY,
                                  point_size=2, render_points_as_spheres=True)

            for si in seg_ids_vis:
                a = segments[si]["a"]
                b = segments[si]["b"]
                ln = pv.Line(a, b)
                ln["dens_vis"] = np.array([dens_vis_all[si]], dtype=float)
                plot_all.add_mesh(ln, scalars="dens_vis", cmap=HEATMAP_CMAP,
                                  clim=(lo, hi), line_width=NONSELECT_LINE_W,
                                  render_lines_as_tubes=True, show_scalar_bar=False)

            # plot_all.add_scalar_bar(title="Point density", vertical=True,  #plot_all.add_scalar_bar(title=f"{DENSITY_VIS_MODE}(density_for_filter)", vertical=False,
            #                         position_x=0.88, position_y=0.20, width=0.06, height=0.60, title_font_size=36,label_font_size=32)
            # plot_all.show_axes()

            plot_all.reset_camera()
            if PREFILTER_CAMERA_ISO:
                plot_all.camera_position = "iso"
            try:
                plot_all.camera.zoom(float(PREFILTER_CAMERA_ZOOM))
            except Exception:
                pass

            if OFFSCREEN_RENDER:
                plot_all.show(screenshot=OUT_PNG_ALL, auto_close=True, interactive=False)
            else:
                plot_all.show(screenshot=OUT_PNG_ALL, auto_close=True)

            info(f"Saved prefilter PNG -> {OUT_PNG_ALL} (plotted {len(seg_ids_vis)}/{len(seg_ids_all)} segments)")

        except Exception as e:
            warn(f"Could not save prefilter screenshot: {e}")

    # 5b) Option A - filter segments ----------------------------------------------------
    if FILTER_MODE == "top_pct":
        k = max(1, int(math.ceil(len(segments) * (TOP_PCT / 100.0))))
        order = np.argsort(density_for_filter)[::-1]
        selected_idx = set(order[:k].tolist())

        metric_name = "density_norm" if USE_NORMALIZED_DENSITY_FOR_FILTER else "density"
        info(f"[FILTER] Keeping top {TOP_PCT}% by {metric_name} => {k}/{len(segments)}")

    elif FILTER_MODE == "threshold":
        if USE_NORMALIZED_DENSITY_FOR_FILTER:
            thr = float(DENSITY_NORM_THRESH)
            selected_idx = set(np.where(density_for_filter >= thr)[0].tolist())
            info(f"[FILTER] Keeping density_norm >= {thr:.3f} => {len(selected_idx)}/{len(segments)}")
        else:
            thr = float(DENSITY_THRESH)
            selected_idx = set(np.where(density_for_filter >= thr)[0].tolist())
            info(f"[FILTER] Keeping density >= {thr:.3f} => {len(selected_idx)}/{len(segments)}")
    else:
        raise ValueError("Unknown FILTER_MODE: %s" % FILTER_MODE)

    is_selected = np.array([(i in selected_idx) for i in range(len(segments))], dtype=bool)
    selected_segments = [si for si in range(len(segments)) if is_selected[si]]

    # Build per-plane polygons from FILTERED segments
    plane_segs = defaultdict(list)
    for si in selected_segments:
        s = segments[si]
        i, j = s["pair"]
        plane_segs[i].append((s["a"], s["b"]))
        plane_segs[j].append((s["a"], s["b"]))

    plane_polygons = {}
    for lab in plane_labels:
        segs_lab = plane_segs.get(lab, [])
        if len(segs_lab) < 3:
            continue
        poly = plane_polygons_from_segments(lab, plane_info[lab], segs_lab)
        if poly is None:
            continue
        plane_polygons[lab] = poly

    info(f"Planes with polygons from filtered segments: {len(plane_polygons)}/{len(plane_labels)}")

    # -------------------------------------------------------------------------------------
    # NEW: Visualization right after polygon formation
    # -------------------------------------------------------------------------------------
    if SAVE_POLYGON_PNG:
        try:
            plot_poly = pv.Plotter(off_screen=OFFSCREEN_RENDER, window_size=(1600, 1000))

            center = 0.5 * (bbox_min + bbox_max)
            diag = float(np.linalg.norm(bbox_max - bbox_min))

            # segments under polygons (optional)
            if POLY_DRAW_SELECTED_SEGS:
                sel_ids = [i for i in range(len(segments)) if is_selected[i]]
                for si in sel_ids:
                    a = segments[si]["a"]; b = segments[si]["b"]
                    ln = pv.Line(a, b)
                    plot_poly.add_mesh(ln, color="white", opacity=0.25, line_width=2.0)

            if POLY_DRAW_ALL_SEGS_FAINT:
                non_sel_ids = [i for i in range(len(segments)) if not is_selected[i]]
                for si in non_sel_ids:
                    a = segments[si]["a"]; b = segments[si]["b"]
                    ln = pv.Line(a, b)
                    plot_poly.add_mesh(ln, color="lightgray", opacity=0.12, line_width=1.0)

            # polygons: fill + boundary
            for lbl, poly in plane_polygons.items():
                nrm = np.asarray(plane_info[int(lbl)]["normal"], float)

                for loop3d in poly["loops_3d"]:
                    pts = np.asarray(loop3d, float)
                    pts_off = pts + float(POLY_OFFSET_ALONG_NORMAL) * nrm[None, :]

                    if POLY_FILL_ENABLE:
                        filled = loop3d_to_filled_polygon(pts_off)
                        if filled is not None:
                            plot_poly.add_mesh(
                                filled, color="deepskyblue",
                                opacity=POLY_FILL_OPACITY, show_edges=False
                            )

                    edge = loop3d_to_closed_polyline(pts_off)
                    if edge is not None:
                        plot_poly.add_mesh(
                            edge, color="deepskyblue",
                            opacity=POLY_EDGE_OPACITY, line_width=float(POLY_EDGE_WIDTH)
                        )

            # plot_poly.show_axes()
            plot_poly.camera_position = "iso"

            if OFFSCREEN_RENDER:
                plot_poly.show(screenshot=OUT_PNG_POLY, auto_close=True, interactive=False)
            else:
                plot_poly.show(screenshot=OUT_PNG_POLY, auto_close=True)

            info(f"Saved polygon visualization -> {OUT_PNG_POLY}")

        except Exception as e:
            warn(f"Could not save polygon visualization: {e}")
        finally:
            try:
                plot_poly.close()
            except Exception:
                pass

    # 7) Save CSV
    if SAVE_POSTFILTER_CSV:
        rows = []
        for si, s in enumerate(segments):
            rows.append({
                "seg_id": si,
                "plane_i": s["pair"][0],
                "plane_j": s["pair"][1],
                "ax": float(s["a"][0]), "ay": float(s["a"][1]), "az": float(s["a"][2]),
                "bx": float(s["b"][0]), "by": float(s["b"][1]), "bz": float(s["b"][2]),
                "length": float(lengths[si]),
                "count": float(counts[si]),
                "density": float(density[si]),
                "density_norm": float(density_norm[si]),
                "is_selected": bool(is_selected[si]),
            })
        import csv
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        info(f"Saved CSV -> {OUT_CSV}")

    # 8) Save JSON (filtered geometry)
    if SAVE_POSTFILTER_JSON:
        planes_out = []
        for lab in plane_labels:
            n = plane_info[lab]["normal"].tolist()
            d = float(plane_info[lab]["d"])
            entry = {"id": int(lab), "normal": n, "d": d, "num_pts": int(plane_info[lab]["num_pts"])}
            if lab in plane_polygons:
                poly = plane_polygons[lab]
                loops_3d = [loop.tolist() for loop in poly["loops_3d"]]
                entry["poly_loops"] = loops_3d
                entry["poly_loop_areas_uv"] = [float(a) for a in poly["loop_areas"]]
                entry["poly_basis"] = poly["basis"]
            planes_out.append(entry)

        seg_out = []
        for si in selected_segments:
            s = segments[si]
            seg_out.append({
                "seg_id": int(si),
                "pair": [int(s["pair"][0]), int(s["pair"][1])],
                "a": s["a"].tolist(),
                "b": s["b"].tolist(),
                "length": float(lengths[si]),
                "count": float(counts[si]),
                "density": float(density[si]),
                "density_norm": float(density_norm[si]),
            })

        out = {
            "config": {
                "MAX_SNAP_DIST": MAX_SNAP_DIST,
                "PLANE_DIST_EPS": PLANE_DIST_EPS,
                "FILTER_MODE": FILTER_MODE,
                "TOP_PCT": TOP_PCT,
                "DENSITY_THRESH": float(DENSITY_THRESH),
                "DENSITY_NORM_THRESH": float(DENSITY_NORM_THRESH),
                "USE_NORMALIZED_DENSITY_FOR_FILTER": bool(USE_NORMALIZED_DENSITY_FOR_FILTER),
                "DENSITY_VIS_MODE": DENSITY_VIS_MODE,
                "LENGTH_EPS_FOR_DENSITY": float(LENGTH_EPS_FOR_DENSITY),
            },
            "planes": planes_out,
            "selected_segments": seg_out,
        }
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, indent=2)
        info(f"Saved JSON -> {OUT_JSON}")

    # 10) PyRevit JSON EXPORT (AFTER filtering): label, normal, polygon corners
    if SAVE_PYREVIT_JSON:
        export = {
            "meta": {
                "num_planes": int(len(plane_polygons)),
                "method": "PCA TLS plane fitting + density-filtered intersection segments -> per-plane polygon loops (no MBR rectangle)"
            },
            "planes": []
        }

        for lbl, poly in plane_polygons.items():
            nrm = plane_info[int(lbl)]["normal"]
            loops3d = poly["loops_3d"]
            if (loops3d is None) or (len(loops3d) == 0):
                continue

            outer = loops3d[0]
            corners_rounded = [round_vec3(pt, nd=3) for pt in outer]

            rec = {
                "label": int(lbl),
                "normal": nrm.tolist(),
                "corners": corners_rounded,
            }
            export["planes"].append(rec)

        with open(OUT_PYREVIT_JSON, "w") as f:
            json.dump(export, f, indent=2)
        info(f"Saved PyRevit JSON -> {OUT_PYREVIT_JSON}")

    # 9) Visualization (selected colored; non-selected faint)
    plotter = pv.Plotter(off_screen=OFFSCREEN_RENDER, window_size=(1400, 900))

    center = 0.5 * (bbox_min + bbox_max)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    #todo: plane visual

    # for lab in plane_labels:
    #     n = plane_info[lab]["normal"]
    #     d = plane_info[lab]["d"]
    #     t = -(np.dot(n, center) + d)
    #     p0 = center + t * n
    #     plane = pv.Plane(center=p0, direction=n, i_size=diag, j_size=diag)
    #     plotter.add_mesh(plane, color="tan", opacity=PLANE_OPACITY, show_edges=False)

    if noisy_pts.shape[0] > 0:
        cloud = pv.PolyData(noisy_pts)
        plotter.add_mesh(cloud, color="red", opacity=NOISY_OPACITY,
                         render_points_as_spheres=True, point_size=NOISY_POINT_SIZE)

    non_sel_ids = [i for i in range(len(segments)) if not is_selected[i]]
    if non_sel_ids:
        pts = []
        lines = []
        pid = 0
        for si in non_sel_ids:
            a = segments[si]["a"]
            b = segments[si]["b"]
            pts.append(a)
            pts.append(b)
            lines.append([2, pid, pid + 1])
            pid += 2
        poly = pv.PolyData(np.array(pts))
        poly.lines = np.array(lines, dtype=np.int64)
        plotter.add_mesh(poly, color="lightgray", opacity=NONSELECT_OPACITY, line_width=NONSELECT_LINE_W)

    sel_ids = [i for i in range(len(segments)) if is_selected[i]]
    if sel_ids:
        dens_sel = density_for_filter[sel_ids]
        dens_vis = density_transform(dens_sel, DENSITY_VIS_MODE)

        if DENSITY_CLIP_PCT is not None and len(dens_vis) >= 3:
            lo, hi = np.percentile(dens_vis, DENSITY_CLIP_PCT)
            if hi <= lo + 1e-12:
                lo, hi = float(dens_vis.min()), float(dens_vis.max() + 1e-9)
        else:
            lo, hi = float(dens_vis.min()), float(dens_vis.max() + 1e-9)

        if float(dens_vis.max()) > float(dens_vis.min()) + 1e-12:
            w_norm = (dens_vis - dens_vis.min()) / (dens_vis.max() - dens_vis.min())
        else:
            w_norm = np.zeros_like(dens_vis)
        widths = SELECT_LINE_W_MIN + w_norm * (SELECT_LINE_W_MAX - SELECT_LINE_W_MIN)

        for idx_local, si in enumerate(sel_ids):
            a = segments[si]["a"]
            b = segments[si]["b"]
            poly = pv.Line(a, b)
            poly["dens_vis"] = np.array([dens_vis[idx_local]])
            plotter.add_mesh(poly, scalars="dens_vis", cmap=HEATMAP_CMAP,
                             clim=(lo, hi), line_width=float(widths[idx_local]),
                             show_scalar_bar=False)

    #     plotter.add_scalar_bar(title="Point density", vertical=True, #plotter.add_scalar_bar(title=f"{DENSITY_VIS_MODE}(density_for_filter)", vertical=False,
    #                            position_x=0.88, position_y=0.20, width=0.06, height=0.60,title_font_size=36,label_font_size=32)
    #
    # plotter.show_axes()
    plotter.camera_position = "iso"

    try:
        if SAVE_POSTFILTER_PNG:
            if OFFSCREEN_RENDER:
                plotter.show(screenshot=OUT_PNG, auto_close=True, interactive=False)
            else:
                plotter.show(screenshot=OUT_PNG, auto_close=True)
            info(f"Saved screenshot -> {OUT_PNG}")
        else:
            if OFFSCREEN_RENDER:
                plotter.show(auto_close=True, interactive=False)
            else:
                plotter.show(auto_close=True)
    except Exception as e:
        warn(f"Could not save screenshot: {e}")
    finally:
        try:
            plotter.close()
        except Exception:
            pass

    info("[DONE] Option A segment filtering complete.")


if __name__ == "__main__":
    main()
