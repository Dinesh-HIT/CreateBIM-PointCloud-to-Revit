"""
polygon_bim_density.py
======================
Core geometry pipeline (faithful to reference script):
  1.  PCA/TLS plane fit per cluster label
  2.  Triple-plane LLS intersection → candidate corner vertices
  3.  Pair-wise segments from vertices  (plane-plane intersection lines,
      endpoints constrained by triple-plane intersections)
  4.  Noisy-boundary-point voting → normalised density per segment
  5.  Density filter  (threshold or top-pct)
  6.  Per-plane PSLG polygon tracing  (UV projection → CCW loop)
  7.  PyRevit JSON export  +  full geometry JSON  +  CSV  +  PNGs

Outputs  (all written to OUT_DIR)
---------
  pyrevit_planes.json          ← corners per plane for Revit Wall/Slab
  geometry_output.json         ← full detail (planes + selected segments)
  segments_density.csv         ← per-segment density table
  view_density_map_all.png     ← pre-filter heat-map
  view_density_map_filtered.png← post-filter coloured segments
  view_polygons.png            ← final filled polygons

Revit usage
-----------
  for plane in data["planes"]:
      pts  = [XYZ(*c) for c in plane["corners"]]
      loop = CurveLoop()
      for i in range(len(pts)):
          loop.Append(Line.CreateBound(pts[i], pts[(i+1)%len(pts)]))
      # use loop for Floor.Create / Wall face / DirectShape
"""

import os, json, math, csv, bisect
from itertools import combinations
from collections import defaultdict

import numpy as np
import pyvista as pv

# =============================================================================
# CONFIG  ← EDIT THESE PATHS BEFORE RUNNING
# =============================================================================
NPY_PATH  = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_points_clusters_transfer.npy"
NPY_NOISY = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_noisy_points_transfer.npy"
OUT_DIR   = r"F:\CreateBIM\dataset\Output\dbscan\density_map_F\experiment_1_voxel_new"

# ---------- Geometry / robustness -------------------------------------------
BBOX_PAD              = 0.50
BBOX_ROBUST_PCT       = (0.5, 99.5)   # percentile clip for bounding box
PLANE_DIST_EPS        = 0.05          # max distance: vertex to its 3 planes (m)
DET_EPS               = 1e-10         # reject near-degenerate triplets
MIN_SEG_LEN           = 1e-3          # drop segments shorter than this (m)

# ---------- Vertex refinement (optional) ------------------------------------
VERTEX_REFINE_ENABLE       = False    # True → cluster + LLS refinement
CORNER_CLUSTER_RADIUS      = 0.10    # m  (only if VERTEX_REFINE_ENABLE=True)
VERTEX_LLS_RESIDUAL_THRESH = None    # e.g. 1e-5 to reject high-residual pts

# ---------- Noisy-point voting ----------------------------------------------
MAX_SNAP_DIST        = 0.15    # gate: ignore noisy pts farther than this (m)
USE_DISTANCE_WEIGHT  = False   # True → Gaussian weight instead of binary vote
SIGMA_DIST           = 0.20    # σ for Gaussian weight (m)
LENGTH_EPS_FOR_DENSITY = 0.05  # avoid divide-by-zero for very short segments

# ---------- Segment filtering -----------------------------------------------
FILTER_MODE                       = "threshold"  # "threshold" or "top_pct"
USE_NORMALIZED_DENSITY_FOR_FILTER = True         # True → filter on [0,1] norm
DENSITY_NORM_THRESH               = 0.1        # used when normalised=True
DENSITY_THRESH                    = 2.0          # used when normalised=False
TOP_PCT                           = 65           # used when FILTER_MODE="top_pct"

# ---------- PSLG polygon parameters -----------------------------------------
POLY_SNAP_EPS_UV      = 0.05    # node snapping grid size (m)
POLY_MIN_EDGE_LEN_UV  = 0.03    # drop edges shorter than this after snapping
POLY_MIN_LOOP_AREA_UV = 0.20    # discard loops with |area| < this (m²)
POLY_COLLINEAR_EPS    = 1e-3    # remove near-collinear vertices (rad)
POLY_MAX_VERTS        = 2000    # safety cap per traced loop

# ---------- Visualisation ---------------------------------------------------
OFFSCREEN_RENDER      = True
DENSITY_VIS_MODE      = "linear"   # "linear", "log1p", or "rank"
DENSITY_CLIP_PCT      = (5, 95)
HEATMAP_CMAP          = "turbo"
NOISY_OPACITY         = 0.25
NOISY_POINT_SIZE      = 2.0
NONSELECT_OPACITY     = 0.15
NONSELECT_LINE_W      = 2.0
SELECT_LINE_W_MIN     = 4.0
SELECT_LINE_W_MAX     = 10.0
POLY_EDGE_WIDTH       = 8.0
POLY_FILL_OPACITY     = 0.18
POLY_OFFSET_NORMAL    = 0.002   # slight normal offset to avoid z-fighting

# ---------- Output toggles --------------------------------------------------
SAVE_PREFILTER_PNG    = True
SAVE_POSTFILTER_CSV   = True
SAVE_POSTFILTER_JSON  = True
SAVE_POSTFILTER_PNG   = True
SAVE_POLYGON_PNG      = True
SAVE_PYREVIT_JSON     = True

# =============================================================================
# Output paths  (auto-derived)
# =============================================================================
OUT_CSV          = os.path.join(OUT_DIR, "segments_density.csv")
OUT_JSON         = os.path.join(OUT_DIR, "geometry_output.json")
OUT_PYREVIT      = os.path.join(OUT_DIR, "pyrevit_planes.json")
OUT_PNG_ALL      = os.path.join(OUT_DIR, "view_density_map_all.png")
OUT_PNG_FILTERED = os.path.join(OUT_DIR, "view_density_map_filtered.png")
OUT_PNG_POLY     = os.path.join(OUT_DIR, "view_polygons.png")


# =============================================================================
# Low-level helpers
# =============================================================================
def info(m): print("[INFO]", m)
def warn(m): print("[WARN]", m)

def normalize(v):
    n = float(np.linalg.norm(v))
    return (v / n, n) if n > 1e-12 else (v, 0.0)

def plane_fit_pca_tls(pts):
    """PCA total-least-squares plane: returns (centroid, unit_normal, d)  n·x+d=0."""
    c = pts.mean(0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n, _ = normalize(vh[-1])
    return c, n, -float(np.dot(n, c))

def point_plane_abs(n, d, p):
    return abs(float(np.dot(n, p) + d))

def segment_point_distance(p, a, b):
    """Closest distance from point p to segment [a,b]; also returns t ∈ [0,1]."""
    ab  = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-18:
        return float(np.linalg.norm(p - a)), 0.0
    t = max(0.0, min(1.0, float(np.dot(p - a, ab) / ab2)))
    return float(np.linalg.norm(p - a - t * ab)), t

def unique_points(pts, eps=1e-5):
    if len(pts) == 0:
        return np.empty((0, 3), float)
    arr  = np.asarray(pts, float)
    keys = np.round(arr / eps).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return arr[np.sort(idx)]

def density_transform(vals, mode):
    vals = np.asarray(vals, float)
    if mode == "linear":  return vals
    if mode == "log1p":   return np.log1p(np.clip(vals, 0, None))
    if mode == "rank":
        order = np.argsort(vals)
        r = np.empty_like(order, float)
        r[order] = np.linspace(0, 1, len(vals))
        return r
    raise ValueError(f"Unknown DENSITY_VIS_MODE: {mode}")

def r3(pt, nd=3):
    return [round(float(x), nd) for x in pt]


# =============================================================================
# Triple-plane LLS vertex
# =============================================================================
def find_lls_vertex(normals, offsets, det_eps=None, res_thresh=None):
    """
    Least-squares intersection of m planes (m≥3).
    Returns 3-D point or None if degenerate / residual too large.
    """
    A = np.vstack(normals).astype(float)
    b = -np.array(offsets, float)
    if det_eps is not None and A.shape[0] == 3:
        if abs(float(np.linalg.det(A))) < det_eps:
            return None
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(x)):
        return None
    if res_thresh is not None:
        r = A @ x - b
        if float(np.dot(r, r)) > res_thresh:
            return None
    return x


# =============================================================================
# PSLG polygon tracer
# =============================================================================
def choose_plane_basis(n, pts3d):
    """Stable in-plane (u,v) basis via PCA of projected points."""
    n, _ = normalize(n)
    if pts3d.shape[0] >= 2:
        c = pts3d.mean(0)
        X = pts3d - c
        X = X - (X @ n)[:, None] * n
        if np.linalg.norm(X) > 1e-10:
            _, _, vh = np.linalg.svd(X, full_matrices=False)
            u = vh[0] - np.dot(vh[0], n) * n
            u, un = normalize(u)
            if un > 1e-12:
                v, _ = normalize(np.cross(n, u))
                return u, v
    axes = np.eye(3)
    a = axes[int(np.argmin(np.abs(axes @ n)))]
    u, _ = normalize(np.cross(n, a))
    v, _ = normalize(np.cross(n, u))
    return u, v

def _poly_area_uv(uv):
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def _remove_collinear(uv):
    uv = np.asarray(uv, float)
    if uv.shape[0] < 4:
        return uv
    if np.linalg.norm(uv[0] - uv[-1]) < 1e-12:
        uv = uv[:-1]
    changed = True
    while changed and uv.shape[0] >= 3:
        changed = False
        keep = []
        n = uv.shape[0]
        for i in range(n):
            pp, p, pn = uv[(i-1)%n], uv[i], uv[(i+1)%n]
            v1, v2 = p-pp, pn-p
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 < POLY_MIN_EDGE_LEN_UV or l2 < POLY_MIN_EDGE_LEN_UV:
                changed = True; continue
            c = max(-1., min(1., float(np.dot(v1,v2)/(l1*l2))))
            ang = math.acos(c)
            if ang < POLY_COLLINEAR_EPS or abs(ang - math.pi) < POLY_COLLINEAR_EPS:
                changed = True; continue
            keep.append(p)
        uv = np.asarray(keep, float) if keep else np.empty((0,2))
    return uv

def plane_polygon_from_segments(plane_inf, segs_3d):
    """
    Build the outer CCW polygon for one plane from density-filtered 3-D segments.
    Returns dict with keys: loops_uv, loops_3d, loop_areas, basis.
    """
    n = np.asarray(plane_inf["normal"], float)
    d = float(plane_inf["d"])

    pts3 = np.array([pt for seg in segs_3d for pt in seg], float).reshape(-1, 3)
    if pts3.shape[0] < 6:
        return None

    # Plane origin
    c  = pts3.mean(0)
    p0 = c - (np.dot(n, c) + d) * n
    u, v = choose_plane_basis(n, pts3)

    # Project to UV
    rel   = pts3 - p0
    uv_all = np.column_stack([rel @ u, rel @ v])

    # ── Node snapping ────────────────────────────────────────────────────────
    eps  = float(POLY_SNAP_EPS_UV)
    keys = np.round(uv_all / eps).astype(np.int64)
    key_to_id, node_sum, node_cnt = {}, [], []

    def node_id(k, uv_pt):
        k = (int(k[0]), int(k[1]))
        if k in key_to_id:
            nid = key_to_id[k]
            node_sum[nid] += uv_pt; node_cnt[nid] += 1
            return nid
        nid = len(node_sum)
        key_to_id[k] = nid
        node_sum.append(np.array(uv_pt, float)); node_cnt.append(1)
        return nid

    node_ids = [node_id(keys[i], uv_all[i]) for i in range(uv_all.shape[0])]
    node_uv  = np.array([node_sum[i] / node_cnt[i] for i in range(len(node_sum))], float)

    # ── Build edges ──────────────────────────────────────────────────────────
    edges = set()
    for e in range(0, len(node_ids), 2):
        a, b = node_ids[e], node_ids[e+1]
        if a == b: continue
        if np.linalg.norm(node_uv[b] - node_uv[a]) < POLY_MIN_EDGE_LEN_UV: continue
        edges.add((a, b) if a < b else (b, a))
    if len(edges) < 3:
        return None

    # ── Adjacency sorted by angle ─────────────────────────────────────────
    adj = defaultdict(list)
    for a, b in edges:
        ua, va = node_uv[a]; ub, vb = node_uv[b]
        adj[a].append((b, math.atan2(vb-va, ub-ua)))
        adj[b].append((a, math.atan2(va-vb, ua-ub)))
    for k in adj:
        adj[k].sort(key=lambda t: t[1])

    def next_ccw(curr_b, ang_in):
        outs = adj[curr_b]
        if not outs: return None
        angs = [x[1] for x in outs]
        idx  = bisect.bisect_right(angs, ang_in) % len(outs)
        return outs[idx]

    # ── Trace CCW cycles ─────────────────────────────────────────────────
    visited = set()
    cycles  = []

    for start_a in list(adj.keys()):
        for start_b, _ in list(adj[start_a]):
            if (start_a, start_b) in visited:
                continue
            loop = []
            ca, cb = start_a, start_b
            for _ in range(POLY_MAX_VERTS):
                visited.add((ca, cb))
                loop.append(ca)
                ua, va = node_uv[ca]; ub, vb = node_uv[cb]
                nxt = next_ccw(cb, math.atan2(va-vb, ua-ub))
                if nxt is None: break
                nc, _ = nxt
                ca, cb = cb, nc
                if (ca, cb) == (start_a, start_b): break

            uniq = list(dict.fromkeys(loop))
            if len(uniq) < 3: continue

            uv_lp = np.array([node_uv[i] for i in uniq], float)
            if np.linalg.norm(uv_lp[0] - uv_lp[-1]) < 1e-12:
                uv_lp = uv_lp[:-1]
            if uv_lp.shape[0] < 3: continue

            uv_lp = _remove_collinear(uv_lp)
            if uv_lp.shape[0] < 3: continue

            area = _poly_area_uv(uv_lp)
            if abs(area) < POLY_MIN_LOOP_AREA_UV: continue

            # Canonical key for dedup
            coords = [tuple(np.round(p, 6)) for p in uv_lp]
            best = None
            for seq in (coords, list(reversed(coords))):
                mi = min(range(len(seq)), key=lambda i: seq[i])
                rot = tuple(seq[mi:] + seq[:mi])
                if best is None or rot < best: best = rot
            cycles.append((best, uv_lp, area))

    if not cycles: return None

    by_key = {}
    for key, uv_lp, area in cycles:
        if key not in by_key or abs(area) > abs(by_key[key][1]):
            by_key[key] = (uv_lp, area)

    loops = sorted(by_key.values(), key=lambda t: -abs(t[1]))
    outer_uv, outer_area = loops[0]

    # Enforce CCW
    if outer_area < 0:
        outer_uv = outer_uv[::-1]

    outer_3d = p0 + outer_uv[:, 0:1] * u + outer_uv[:, 1:2] * v

    return {
        "basis"      : {"origin": p0.tolist(), "u": u.tolist(), "v": v.tolist()},
        "loops_uv"   : [outer_uv],
        "loops_3d"   : [outer_3d],
        "loop_areas" : [float(abs(outer_area))],
    }


# =============================================================================
# PyVista mesh helpers
# =============================================================================
def make_closed_polyline(loop3d):
    pts = np.asarray(loop3d, float); nn = pts.shape[0]
    if nn < 2: return None
    lines = np.array([[2,i,i+1] for i in range(nn-1)]+[[2,nn-1,0]], np.int64)
    pd = pv.PolyData(pts); pd.lines = lines
    return pd

def make_filled_polygon(loop3d):
    pts = np.asarray(loop3d, float); nn = pts.shape[0]
    if nn < 3: return None
    faces = np.hstack([[nn], np.arange(nn, dtype=np.int64)]).astype(np.int64)
    pd = pv.PolyData(pts, faces=faces)
    try: pd = pd.triangulate()
    except Exception: pass
    return pd


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    info(f"Loading clusters : {NPY_PATH}")
    raw = np.load(NPY_PATH)
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError("Expected (N,4+): x,y,z,label")
    pts_all, labels_all = raw[:,:3].astype(float), raw[:,3].astype(int)

    info(f"Loading noisy pts: {NPY_NOISY}")
    noisy_pts = np.load(NPY_NOISY)[:,:3].astype(float)
    info(f"Cluster pts={len(pts_all)}  Noisy pts={len(noisy_pts)}")

    # ── 2. Bounding box ───────────────────────────────────────────────────────
    bb_all   = np.vstack([pts_all, noisy_pts]) if noisy_pts.shape[0] else pts_all
    bbox_min = np.percentile(bb_all, BBOX_ROBUST_PCT[0], axis=0) - BBOX_PAD
    bbox_max = np.percentile(bb_all, BBOX_ROBUST_PCT[1], axis=0) + BBOX_PAD

    # ── 3. Plane fitting ──────────────────────────────────────────────────────
    plane_info = {}
    for lab in np.unique(labels_all):
        pts = pts_all[labels_all == lab]
        if pts.shape[0] < 10:
            warn(f"  Cluster {lab}: only {pts.shape[0]} pts – skipped"); continue
        c, n, d = plane_fit_pca_tls(pts)
        plane_info[int(lab)] = dict(centroid=c, normal=n, d=d, num_pts=int(pts.shape[0]))
    plane_labels = sorted(plane_info)
    info(f"Fitted planes: {len(plane_labels)}")

    # ── 4. Triple-plane vertices → candidate segment endpoints ────────────────
    raw_verts = []
    for i, j, k in combinations(plane_labels, 3):
        ns = [plane_info[x]["normal"] for x in (i,j,k)]
        ds = [plane_info[x]["d"]      for x in (i,j,k)]
        v  = find_lls_vertex(ns, ds,
                              det_eps   = DET_EPS,
                              res_thresh= VERTEX_LLS_RESIDUAL_THRESH)
        if v is None: continue
        if np.any(v < bbox_min) or np.any(v > bbox_max): continue
        if any(point_plane_abs(ns[t], ds[t], v) > PLANE_DIST_EPS for t in range(3)): continue
        raw_verts.append({"point": v, "planes": {i, j, k}})

    if not VERTEX_REFINE_ENABLE:
        vertices = [(rv["point"], tuple(sorted(rv["planes"]))) for rv in raw_verts]
    else:
        # Cluster and re-solve with all contributing planes
        clusters_v = []
        r2 = CORNER_CLUSTER_RADIUS ** 2
        for rv in raw_verts:
            p = rv["point"]
            placed = False
            for cl in clusters_v:
                if float(np.sum((p - cl["center"])**2)) <= r2:
                    cl["pts"].append(p); cl["planes"] |= rv["planes"]
                    cl["center"] = np.mean(cl["pts"], 0); placed=True; break
            if not placed:
                clusters_v.append({"pts":[p], "planes":set(rv["planes"]), "center":p.copy()})
        vertices = []
        for cl in clusters_v:
            if len(cl["planes"]) < 3: continue
            ns = [plane_info[l]["normal"] for l in cl["planes"]]
            ds = [plane_info[l]["d"]      for l in cl["planes"]]
            v  = find_lls_vertex(ns, ds, res_thresh=VERTEX_LLS_RESIDUAL_THRESH)
            if v is None: continue
            if np.any(v < bbox_min) or np.any(v > bbox_max): continue
            if any(point_plane_abs(ns[t], ds[t], v) > PLANE_DIST_EPS
                   for t in range(len(ns))): continue
            vertices.append((v, tuple(sorted(cl["planes"]))))

    info(f"Candidate vertices: {len(vertices)}")

    # ── 5. Build intersection segments ───────────────────────────────────────
    pair_to_pts = defaultdict(list)
    for v, planes in vertices:
        for a, b in combinations(list(planes), 2):
            pair_to_pts[tuple(sorted((a,b)))].append(v)

    segments = []
    for i, j in combinations(plane_labels, 2):
        pair = tuple(sorted((i,j)))
        pts  = pair_to_pts.get(pair, [])
        if len(pts) < 2: continue
        pts_u = unique_points(pts)
        if pts_u.shape[0] < 2: continue
        dirv, dn = normalize(np.cross(plane_info[i]["normal"],
                                       plane_info[j]["normal"]))
        if dn < 1e-10: continue
        order = np.argsort(pts_u @ dirv)
        for a, b in zip(pts_u[order[:-1]], pts_u[order[1:]]):
            L = float(np.linalg.norm(b - a))
            if L < MIN_SEG_LEN: continue
            segments.append({"pair": pair, "a": a, "b": b, "length": L})
    info(f"Intersection segments: {len(segments)}")
    if not segments:
        warn("No segments built – check PLANE_DIST_EPS / clustering."); return

    # ── 6. Noisy-point boundary voting ───────────────────────────────────────
    counts = np.zeros(len(segments), float)
    for p in noisy_pts:
        best_i, best_d = -1, 1e9
        for si, s in enumerate(segments):
            d, _ = segment_point_distance(p, s["a"], s["b"])
            if d < best_d: best_d = d; best_i = si
        if best_i >= 0 and best_d <= MAX_SNAP_DIST:
            if USE_DISTANCE_WEIGHT:
                counts[best_i] += math.exp(-(best_d**2) / (2*SIGMA_DIST**2))
            else:
                counts[best_i] += 1.0

    lengths      = np.array([s["length"] for s in segments], float)
    density      = counts / np.maximum(lengths, LENGTH_EPS_FOR_DENSITY)
    max_d        = float(density.max()) if density.size else 0.
    density_norm = density / max_d if max_d > 1e-12 else np.zeros_like(density)
    density_flt  = density_norm if USE_NORMALIZED_DENSITY_FOR_FILTER else density

    # ── 7. Pre-filter density PNG ─────────────────────────────────────────────
    if SAVE_PREFILTER_PNG:
        try:
            dv = density_transform(density_flt, DENSITY_VIS_MODE)
            lo, hi = np.percentile(dv, (10, 95))
            if hi <= lo + 1e-12: hi = float(dv.max()) + 1e-9
            pl = pv.Plotter(off_screen=OFFSCREEN_RENDER, window_size=(1400,900))
            if noisy_pts.shape[0]:
                pl.add_mesh(pv.PolyData(noisy_pts), color="red",
                            opacity=NOISY_OPACITY, point_size=NOISY_POINT_SIZE,
                            render_points_as_spheres=True)
            for si, s in enumerate(segments):
                ln = pv.Line(s["a"], s["b"])
                ln["dv"] = np.array([dv[si]], float)
                pl.add_mesh(ln, scalars="dv", cmap=HEATMAP_CMAP,
                            clim=(lo,hi), line_width=NONSELECT_LINE_W,
                            render_lines_as_tubes=True, show_scalar_bar=False)
            pl.camera_position = "iso"
            pl.show(screenshot=OUT_PNG_ALL, auto_close=True, interactive=False)
            info(f"Saved pre-filter PNG → {OUT_PNG_ALL}")
        except Exception as e:
            warn(f"Pre-filter PNG failed: {e}")

    # ── 8. Filter segments ────────────────────────────────────────────────────
    if FILTER_MODE == "top_pct":
        k = max(1, int(math.ceil(len(segments) * TOP_PCT / 100.)))
        sel_idx = set(np.argsort(density_flt)[::-1][:k].tolist())
        info(f"Filter top-{TOP_PCT}% → {len(sel_idx)}/{len(segments)}")
    elif FILTER_MODE == "threshold":
        thr = DENSITY_NORM_THRESH if USE_NORMALIZED_DENSITY_FOR_FILTER else DENSITY_THRESH
        sel_idx = set(np.where(density_flt >= thr)[0].tolist())
        info(f"Filter density_{'norm' if USE_NORMALIZED_DENSITY_FOR_FILTER else 'raw'}"
             f" >= {thr} → {len(sel_idx)}/{len(segments)}")
    else:
        raise ValueError(FILTER_MODE)

    is_sel   = np.array([i in sel_idx for i in range(len(segments))], bool)
    sel_segs = [si for si in range(len(segments)) if is_sel[si]]

    # ── 9. Post-filter density PNG ────────────────────────────────────────────
    if SAVE_POSTFILTER_PNG:
        try:
            dv_s = density_transform(density_flt[sel_segs], DENSITY_VIS_MODE)
            if len(dv_s) >= 3:
                lo_s, hi_s = np.percentile(dv_s, DENSITY_CLIP_PCT)
            else:
                lo_s, hi_s = float(dv_s.min()), float(dv_s.max()+1e-9)
            if hi_s <= lo_s + 1e-12: hi_s = float(dv_s.max()) + 1e-9
            wn = (dv_s - dv_s.min()) / max(dv_s.max()-dv_s.min(), 1e-12)
            widths = SELECT_LINE_W_MIN + wn*(SELECT_LINE_W_MAX-SELECT_LINE_W_MIN)

            pl2 = pv.Plotter(off_screen=OFFSCREEN_RENDER, window_size=(1400,900))
            if noisy_pts.shape[0]:
                pl2.add_mesh(pv.PolyData(noisy_pts), color="red",
                             opacity=NOISY_OPACITY, point_size=NOISY_POINT_SIZE,
                             render_points_as_spheres=True)
            # faint non-selected
            non_sel = [i for i in range(len(segments)) if not is_sel[i]]
            if non_sel:
                pts_ns = np.vstack([[segments[si]["a"], segments[si]["b"]]
                                    for si in non_sel])
                ln_ns  = np.array([[2,2*k,2*k+1] for k in range(len(non_sel))], np.int64)
                pd_ns  = pv.PolyData(pts_ns); pd_ns.lines = ln_ns
                pl2.add_mesh(pd_ns, color="lightgray",
                             opacity=NONSELECT_OPACITY, line_width=NONSELECT_LINE_W)
            # selected heat-map
            for ii, si in enumerate(sel_segs):
                ln = pv.Line(segments[si]["a"], segments[si]["b"])
                ln["dv"] = np.array([dv_s[ii]], float)
                pl2.add_mesh(ln, scalars="dv", cmap=HEATMAP_CMAP,
                             clim=(lo_s,hi_s), line_width=float(widths[ii]),
                             render_lines_as_tubes=True, show_scalar_bar=False)
            pl2.camera_position = "iso"
            pl2.show(screenshot=OUT_PNG_FILTERED, auto_close=True, interactive=False)
            info(f"Saved post-filter PNG → {OUT_PNG_FILTERED}")
        except Exception as e:
            warn(f"Post-filter PNG failed: {e}")

    # ── 10. Per-plane PSLG polygon tracing ───────────────────────────────────
    plane_segs = defaultdict(list)
    for si in sel_segs:
        s = segments[si]
        for pid in s["pair"]:
            plane_segs[pid].append((s["a"], s["b"]))

    plane_polygons = {}
    for lab in plane_labels:
        segs_lab = plane_segs.get(lab, [])
        if len(segs_lab) < 3: continue
        poly = plane_polygon_from_segments(plane_info[lab], segs_lab)
        if poly is None: continue
        plane_polygons[lab] = poly
    info(f"Planes with polygon: {len(plane_polygons)}/{len(plane_labels)}")

    # ── 11. Polygon visualisation PNG ────────────────────────────────────────
    if SAVE_POLYGON_PNG:
        try:
            pl3 = pv.Plotter(off_screen=OFFSCREEN_RENDER, window_size=(1600,1000))
            # selected segments as thin white underlayer
            for si in sel_segs:
                ln = pv.Line(segments[si]["a"], segments[si]["b"])
                pl3.add_mesh(ln, color="white", opacity=0.25, line_width=2.)
            # polygon fill + edge
            for lbl, poly in plane_polygons.items():
                nrm = np.asarray(plane_info[int(lbl)]["normal"], float)
                for loop3d in poly["loops_3d"]:
                    pts_off = np.asarray(loop3d, float) + POLY_OFFSET_NORMAL * nrm
                    filled  = make_filled_polygon(pts_off)
                    if filled:
                        pl3.add_mesh(filled, color="deepskyblue",
                                     opacity=POLY_FILL_OPACITY, show_edges=False)
                    edge = make_closed_polyline(pts_off)
                    if edge:
                        pl3.add_mesh(edge, color="deepskyblue",
                                     opacity=1.0, line_width=float(POLY_EDGE_WIDTH))
            pl3.camera_position = "iso"
            pl3.show(screenshot=OUT_PNG_POLY, auto_close=True, interactive=False)
            info(f"Saved polygon PNG → {OUT_PNG_POLY}")
        except Exception as e:
            warn(f"Polygon PNG failed: {e}")
        finally:
            try: pl3.close()
            except Exception: pass

    # ── 12. CSV ───────────────────────────────────────────────────────────────
    if SAVE_POSTFILTER_CSV:
        rows = []
        for si, s in enumerate(segments):
            rows.append({
                "seg_id"      : si,
                "plane_i"     : s["pair"][0],
                "plane_j"     : s["pair"][1],
                "ax": round(float(s["a"][0]),4), "ay": round(float(s["a"][1]),4),
                "az": round(float(s["a"][2]),4),
                "bx": round(float(s["b"][0]),4), "by": round(float(s["b"][1]),4),
                "bz": round(float(s["b"][2]),4),
                "length"      : round(float(lengths[si]),4),
                "count"       : round(float(counts[si]),2),
                "density"     : round(float(density[si]),4),
                "density_norm": round(float(density_norm[si]),4),
                "is_selected" : bool(is_sel[si]),
            })
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        info(f"Saved CSV → {OUT_CSV}")

    # ── 13. Full geometry JSON ────────────────────────────────────────────────
    if SAVE_POSTFILTER_JSON:
        planes_out = []
        for lab in plane_labels:
            entry = {"id": int(lab),
                     "normal": plane_info[lab]["normal"].tolist(),
                     "d": float(plane_info[lab]["d"]),
                     "num_pts": plane_info[lab]["num_pts"]}
            if lab in plane_polygons:
                poly = plane_polygons[lab]
                entry["poly_loops_3d"]   = [lp.tolist() for lp in poly["loops_3d"]]
                entry["poly_loop_areas"] = [float(a) for a in poly["loop_areas"]]
                entry["poly_basis"]      = poly["basis"]
            planes_out.append(entry)

        seg_out = [{"seg_id": int(si), "pair": list(segments[si]["pair"]),
                    "a": segments[si]["a"].tolist(), "b": segments[si]["b"].tolist(),
                    "length": float(lengths[si]), "count": float(counts[si]),
                    "density": float(density[si]),
                    "density_norm": float(density_norm[si])}
                   for si in sel_segs]

        with open(OUT_JSON, "w") as f:
            json.dump({
                "config": {
                    "MAX_SNAP_DIST": MAX_SNAP_DIST,
                    "PLANE_DIST_EPS": PLANE_DIST_EPS,
                    "FILTER_MODE": FILTER_MODE,
                    "DENSITY_NORM_THRESH": float(DENSITY_NORM_THRESH),
                    "USE_NORMALIZED_DENSITY_FOR_FILTER": USE_NORMALIZED_DENSITY_FOR_FILTER,
                },
                "planes": planes_out,
                "selected_segments": seg_out,
            }, f, indent=2)
        info(f"Saved geometry JSON → {OUT_JSON}")

    # ── 14. PyRevit JSON ──────────────────────────────────────────────────────
    if SAVE_PYREVIT_JSON:
        export = {
            "meta": {
                "num_planes": len(plane_polygons),
                "method": (
                    "PCA-TLS plane fit → triple-plane LLS vertices → "
                    "plane-plane intersection segments → noisy-point density voting → "
                    "threshold filter → per-plane PSLG CCW polygon"
                ),
            },
            "planes": [],
        }
        for lbl, poly in plane_polygons.items():
            loops3d = poly["loops_3d"]
            if not loops3d: continue
            outer    = loops3d[0]   # largest CCW loop
            corners  = [r3(pt) for pt in outer]
            # Close the loop (first == last) as required by Revit CurveLoop
            if corners[0] != corners[-1]:
                corners.append(corners[0])
            export["planes"].append({
                "label"          : int(lbl),
                "normal"         : plane_info[int(lbl)]["normal"].tolist(),
                "d"              : round(float(plane_info[int(lbl)]["d"]), 4),
                "polygon_area_m2": round(float(poly["loop_areas"][0]), 3),
                "num_cluster_pts": plane_info[int(lbl)]["num_pts"],
                "corners_ccw_3d" : corners,
            })
        with open(OUT_PYREVIT, "w") as f:
            json.dump(export, f, indent=2)
        info(f"Saved PyRevit JSON → {OUT_PYREVIT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    info("=" * 55)
    info(f"  Planes fitted      : {len(plane_labels)}")
    info(f"  Candidate vertices : {len(vertices)}")
    info(f"  Segments built     : {len(segments)}")
    info(f"  Segments selected  : {len(sel_segs)}")
    info(f"  Polygons traced    : {len(plane_polygons)}")
    info("[DONE]")


if __name__ == "__main__":
    main()