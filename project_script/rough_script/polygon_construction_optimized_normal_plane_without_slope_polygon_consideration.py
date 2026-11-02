"""
pipeline_full.py
================
Full Point-Cloud → Polygon → PyRevit pipeline

Steps
-----
 1  PCA plane fit per cluster
 2  Compute normal + plane position (centroid projected onto plane)
 3  Normalise plane normal
 4  Snap normal: |n_z| > 0.70 → horizontal (0,0,1)
                 otherwise    → vertical   (n_x,n_y,0)
    Guard: if XY norm < 1e-6 on a wall plane → skip (degenerate)
    Re-project centroid onto updated plane to get consistent position
 5  Build plane dictionary
 6  Triple-plane LLS vertices → plane-pair intersection segments
 7  Noisy-point density voting per segment
 8  Linear density  =  votes / length
 9  Normalise density; apply threshold filter
10  Project filtered segment endpoints to local 2-D (u,v) per plane
11  PSLG (V,E) + CCW traversal → polygon loops
12  Back-project polygon loops to 3-D
13  PyRevit JSON + geometry JSON
14  Figures at 300 DPI — thesis quality (all font sizes consistent):
      Fig-1   Fitted planes        — SINGLE isometric 3-D view
      Fig-2   Candidate segments   — isometric + top-down (3-D)
      Fig-2b  Candidate segments   — top-down 2-D (XY plane)
      Fig-3   Segments by density  — isometric + top-down (3-D, turbo)
      Fig-3b  Segments by density  — top-down 2-D (XY plane, turbo)
      Fig-4   Density heat-map     — isometric + top-down (3-D, width-scaled)
      Fig-4b  Density heat-map     — top-down 2-D (XY plane, width-scaled)
      Fig-5   Filtered segments    — isometric + top-down (3-D)
      Fig-5b  Filtered segments    — top-down 2-D (XY plane)
      Fig-6   Polygon formation    — isometric + top-down (3-D)

NOTE (3-D axis appearance, updated):
    The 3-D axes now draw a clean white bounding box with crisp edges,
    push axis labels and tick numbers clear of the grid lines, and pad the
    data limits so geometry does not touch the box walls — suitable for a
    thesis. All of this is controlled in _style_ax3d / _force_tick_size_3d /
    _expand_limits / _save.  Tune the AX3D_* constants below in one place.
"""

import os, json, math, csv, bisect
from itertools import combinations
from collections import defaultdict

import numpy as np
from numpy.linalg import svd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D                    # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import pyvista as pv

# =============================================================================
# THESIS-GRADE FONT SIZES  (single source of truth — change here only)
# =============================================================================
FS_SUPTITLE  = 22   # figure-level suptitle
FS_SUBTITLE  = 22   # subplot title (isometric / top-down label)
FS_AXLABEL   = 24   # X / Y / Z axis labels
FS_TICK      = 28   # tick numbers on ALL axes (X, Y, Z) and colorbar ticks
FS_LEGEND    = 24   # legend text
FS_CBAR      = 18   # colour-bar label text
FS_ANNOT     = 16   # in-plot annotations (plane labels L0, L1 …)

# =============================================================================
# 3-D AXIS APPEARANCE  (single source of truth for the box / spacing)
# =============================================================================
AX3D_LABELPAD     = 24        # distance of X/Y/Z axis labels from the cube
AX3D_TICKPAD_XY   = 8         # distance of X/Y tick numbers from grid lines
AX3D_TICKPAD_Z    = 10        # distance of Z tick numbers from grid lines
AX3D_Z_LABELPAD_TD = 14       # Z-label pad for steep top-down panels
AX3D_Z_TICKPAD_TD  = 4        # Z-tick pad for steep top-down panels
AX3D_TOPDOWN_ELEV  = 60       # elev >= this is treated as a top-down panel
AX3D_MAX_TICKS     = 6        # max tick numbers per axis (fewer = no bunching)
AX3D_BOX_EDGE     = '#444444' # bounding-box edge colour (dark = visible box)
AX3D_BOX_EDGE_LW  = 1.4       # bounding-box edge line width
AX3D_PANE_FILL    = 'white'   # cube wall fill colour
AX3D_DATA_PAD     = 0.08      # fraction of data span left as margin to walls
AX3D_TITLE_PAD    = 20        # subplot title pad

# ── global rcParams  (base defaults — explicit per-element calls below override) ──
rcParams['font.family']           = 'DejaVu Sans'
rcParams['font.size']             = FS_TICK
rcParams['axes.titlesize']        = FS_SUBTITLE
rcParams['axes.labelsize']        = FS_AXLABEL
rcParams['xtick.labelsize']       = FS_TICK
rcParams['ytick.labelsize']       = FS_TICK
rcParams['legend.fontsize']       = FS_LEGEND
rcParams['legend.title_fontsize'] = FS_LEGEND
rcParams['axes.linewidth']        = 1.6
rcParams['xtick.major.width']     = 1.6
rcParams['ytick.major.width']     = 1.6
rcParams['xtick.major.size']      = 6
rcParams['ytick.major.size']      = 6

# =============================================================================
# CONFIG
# =============================================================================
# NPY_CLUSTERS = r"F:\CreateBIM\dataset\Output\dbscan\d2k_1_original_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_1_original_noisy_points_transfer.npy"

# NPY_CLUSTERS = r"F:\CreateBIM\dataset\Output\dbscan\d2k_2_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_2_noisy_points_transfer.npy"

# NPY_CLUSTERS = r"F:\CreateBIM\dataset\Output\dbscan\d2k_3_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_3_noisy_points_transfer.npy"
#
# NPY_CLUSTERS = r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_points_clusters_transfer.npy"
# NPY_NOISY    = r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_noisy_points_transfer.npy"

# NPY_CLUSTERS = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_points_clusters_transfer.npy"
# NPY_NOISY    = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_noisy_points_transfer.npy"

# NPY_CLUSTERS = r"F:\CreateBIM\dataset\final_dataset\Experiment_2\experiment_2_points_clusters_transfer.npy"
# NPY_NOISY    = r"F:\CreateBIM\dataset\final_dataset\Experiment_2\experiment_2_noisy_points_transfer.npy"

#todo:NewFOlder Setup

# NPY_CLUSTERS = r"G:\CreateBIM\dataset\dataset1\dbscan_results\d2k_1_original_point_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset1\dbscan_results\d2k_1_original_noisy_points_transfer.npy"

# NPY_CLUSTERS = r"G:\CreateBIM\dataset\dataset2\dbscan_results\d2k_2_point_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset2\dbscan_results\d2k_2_noisy_points_transfer.npy"

NPY_CLUSTERS = r"G:\CreateBIM\dataset\dataset3\dbscan_results\d2k_3_point_clusters_transfer.npy"
NPY_NOISY = r"G:\CreateBIM\dataset\dataset3\dbscan_results\d2k_3_noisy_points_transfer.npy"

# NPY_CLUSTERS = r"G:\CreateBIM\dataset\dataset4\dbscan_results\experiment_1_voxel_points_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset4\dbscan_results\experiment_1_voxel_noisy_points_transfer.npy"

# NPY_CLUSTERS  = r"G:\CreateBIM\dataset\dataset5\dbscan_results\experiment_2_points_clusters_transfer.npy"
# NPY_NOISY = r"G:\CreateBIM\dataset\dataset5\dbscan_results\experiment_2_noisy_points_transfer.npy"


OUT_DIR      = r"G:\CreateBIM\dataset\dataset3\polygon_results"

# ── Geometry ──────────────────────────────────────────────────────────────────
BBOX_PAD          = 0.50
BBOX_ROBUST_PCT   = (0.5, 99.5)
PLANE_DIST_EPS    = 0.08
DET_EPS           = 1e-10
MIN_SEG_LEN       = 0.05
HORIZ_NZ_THRESH   = 0.70
WALL_XY_NORM_MIN  = 1e-6

# ── Voting ────────────────────────────────────────────────────────────────────
MAX_SNAP_DIST = 0.20
LENGTH_EPS    = 0.05

# ── Filtering ─────────────────────────────────────────────────────────────────
DENSITY_NORM_THRESH = 0.000
FILTER_MODE         = "threshold"
TOP_PCT             = 15

# ── PSLG ──────────────────────────────────────────────────────────────────────
POLY_SNAP_EPS_UV      = 0.05
POLY_MIN_EDGE_LEN_UV  = 0.03
POLY_MIN_LOOP_AREA_UV = 0.10
POLY_COLLINEAR_EPS    = 1e-3
POLY_MAX_VERTS        = 3000

# ── Figure / output ───────────────────────────────────────────────────────────
DPI  = 300
CMAP = "turbo"

# ── Output paths ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
F_PLANES   = os.path.join(OUT_DIR, "fig1_fitted_planes_3d.png")
F_TRIPLE   = os.path.join(OUT_DIR, "fig2_triple_vertices_segments_3d.png")
F_ALL_SEG  = os.path.join(OUT_DIR, "fig3_all_segments_3d.png")
F_HEATMAP  = os.path.join(OUT_DIR, "fig4_density_heatmap_3d.png")
F_FILTERED = os.path.join(OUT_DIR, "fig5_filtered_segments_3d.png")
F_POLYGON  = os.path.join(OUT_DIR, "fig6_polygons_3d.png")
F_PYREVIT  = os.path.join(OUT_DIR, "pyrevit_planes.json")
F_GEO      = os.path.join(OUT_DIR, "geometry_polygons.json")
F_CSV      = os.path.join(OUT_DIR, "segments_density.csv")
# 2-D companion figures (top-down XY projection)
F_TRIPLE_2D   = os.path.join(OUT_DIR, "fig2b_candidate_segments_2d.png")
F_ALL_SEG_2D  = os.path.join(OUT_DIR, "fig3b_all_segments_2d.png")
F_HEATMAP_2D  = os.path.join(OUT_DIR, "fig4b_density_heatmap_2d.png")
F_FILTERED_2D = os.path.join(OUT_DIR, "fig5b_filtered_segments_2d.png")


# =============================================================================
# Helpers
# =============================================================================
def info(m): print("[INFO]", m)
def warn(m): print("[WARN]", m)

def normalize(v):
    n = np.linalg.norm(v)
    return (v / n, n) if n > 1e-12 else (v, 0.)

def fit_plane_pca(pts):
    c = pts.mean(0)
    _, _, vh = svd(pts - c, full_matrices=False)
    n = vh[-1]; n /= np.linalg.norm(n)
    d = -float(np.dot(n, c))
    return c, n, d

def project_onto_plane(p, n, d):
    return p - (np.dot(n, p) + d) * n

def point_plane_abs(n, d, p):
    return abs(float(np.dot(n, p) + d))

def seg_pt_dist(p, a, b):
    ab  = b - a; ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-18: return float(np.linalg.norm(p - a)), 0.
    t   = max(0., min(1., float(np.dot(p - a, ab) / ab2)))
    return float(np.linalg.norm(p - a - t * ab)), t

def unique_pts(pts, eps=1e-5):
    arr  = np.asarray(pts, float)
    keys = np.round(arr / eps).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return arr[np.sort(idx)]

def choose_basis(n, pts3d):
    n, _ = normalize(n)
    if pts3d.shape[0] >= 2:
        c = pts3d.mean(0); X = pts3d - c
        X = X - (X @ n)[:, None] * n
        if np.linalg.norm(X) > 1e-10:
            _, _, vh = svd(X, full_matrices=False)
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

def poly_area_uv(uv):
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def remove_collinear(uv):
    uv = np.asarray(uv, float)
    if uv.shape[0] < 4: return uv
    if np.linalg.norm(uv[0] - uv[-1]) < 1e-12: uv = uv[:-1]
    changed = True
    while changed and uv.shape[0] >= 3:
        changed = False; keep = []; n = uv.shape[0]
        for i in range(n):
            pp, p, pn = uv[(i - 1) % n], uv[i], uv[(i + 1) % n]
            v1, v2 = p - pp, pn - p
            l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if l1 < POLY_MIN_EDGE_LEN_UV or l2 < POLY_MIN_EDGE_LEN_UV:
                changed = True; continue
            c = max(-1., min(1., float(np.dot(v1, v2) / (l1 * l2))))
            ang = math.acos(c)
            if ang < POLY_COLLINEAR_EPS or abs(ang - math.pi) < POLY_COLLINEAR_EPS:
                changed = True; continue
            keep.append(p)
        uv = np.asarray(keep, float) if keep else np.empty((0, 2))
    return uv

def lls_vertex(normals, offsets, det_eps=None):
    A = np.vstack(normals).astype(float)
    b = -np.array(offsets, float)
    if det_eps is not None and A.shape[0] == 3:
        if abs(float(np.linalg.det(A))) < det_eps: return None
    try:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return x if np.all(np.isfinite(x)) else None

def r3(pt, nd=3):
    return [round(float(x), nd) for x in pt]

def make_filled(loop3d):
    pts = np.asarray(loop3d, float); nn = pts.shape[0]
    if nn < 3: return None
    faces = np.hstack([[nn], np.arange(nn, dtype=np.int64)]).astype(np.int64)
    pd = pv.PolyData(pts, faces=faces)
    try: pd = pd.triangulate()
    except: pass
    return pd

def make_outline(loop3d):
    pts = np.asarray(loop3d, float); nn = pts.shape[0]
    if nn < 2: return None
    lines = np.array([[2, i, i + 1] for i in range(nn - 1)] + [[2, nn - 1, 0]], np.int64)
    pd = pv.PolyData(pts); pd.lines = lines
    return pd


# =============================================================================
# Polygon tracer
# =============================================================================
def plane_polygon(plane_inf, segs_3d):
    n  = np.asarray(plane_inf["normal"], float)
    d  = float(plane_inf["d"])
    pts3 = np.array([pt for seg in segs_3d for pt in seg], float).reshape(-1, 3)
    if pts3.shape[0] < 6: return None

    c   = pts3.mean(0)
    p0  = c - (np.dot(n, c) + d) * n
    u, v = choose_basis(n, pts3)
    rel    = pts3 - p0
    uv_all = np.column_stack([rel @ u, rel @ v])

    eps    = float(POLY_SNAP_EPS_UV)
    keys   = np.round(uv_all / eps).astype(np.int64)
    key2id, nsum, ncnt = {}, {}, {}

    def nid(k, uv_pt):
        k = (int(k[0]), int(k[1]))
        if k in key2id:
            ni = key2id[k]; nsum[ni] += uv_pt; ncnt[ni] += 1; return ni
        ni = len(nsum); key2id[k] = ni
        nsum[ni] = np.array(uv_pt, float); ncnt[ni] = 1; return ni

    node_ids = [nid(keys[i], uv_all[i]) for i in range(uv_all.shape[0])]
    node_uv  = np.array([nsum[i] / ncnt[i] for i in range(len(nsum))], float)

    edges = set()
    for e in range(0, len(node_ids), 2):
        a, b = node_ids[e], node_ids[e + 1]
        if a == b: continue
        if np.linalg.norm(node_uv[b] - node_uv[a]) < POLY_MIN_EDGE_LEN_UV: continue
        edges.add((a, b) if a < b else (b, a))
    if len(edges) < 3: return None

    adj = defaultdict(list)
    for a, b in edges:
        ua, va = node_uv[a]; ub, vb = node_uv[b]
        adj[a].append((b, math.atan2(vb - va, ub - ua)))
        adj[b].append((a, math.atan2(va - vb, ua - ub)))
    for k in adj: adj[k].sort(key=lambda t: t[1])

    def next_ccw(cb, ang_in):
        outs = adj[cb]
        if not outs: return None
        angs = [x[1] for x in outs]
        idx  = bisect.bisect_right(angs, ang_in) % len(outs)
        return outs[idx]

    visited = set(); cycles = []
    for sa in list(adj.keys()):
        for sb, _ in list(adj[sa]):
            if (sa, sb) in visited: continue
            loop = []; ca, cb = sa, sb
            for _ in range(POLY_MAX_VERTS):
                visited.add((ca, cb)); loop.append(ca)
                ua, va = node_uv[ca]; ub, vb = node_uv[cb]
                nxt = next_ccw(cb, math.atan2(va - vb, ua - ub))
                if nxt is None: break
                nc, _ = nxt; ca, cb = cb, nc
                if (ca, cb) == (sa, sb): break
            uniq = list(dict.fromkeys(loop))
            if len(uniq) < 3: continue
            uv_lp = np.array([node_uv[i] for i in uniq], float)
            if np.linalg.norm(uv_lp[0] - uv_lp[-1]) < 1e-12: uv_lp = uv_lp[:-1]
            if uv_lp.shape[0] < 3: continue
            uv_lp = remove_collinear(uv_lp)
            if uv_lp.shape[0] < 3: continue
            area = poly_area_uv(uv_lp)
            if abs(area) < POLY_MIN_LOOP_AREA_UV: continue
            coords = [tuple(np.round(p, 6)) for p in uv_lp]
            best = None
            for seq in (coords, list(reversed(coords))):
                mi  = min(range(len(seq)), key=lambda i: seq[i])
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
    if outer_area < 0: outer_uv = outer_uv[::-1]
    outer_3d = p0 + outer_uv[:, 0:1] * u + outer_uv[:, 1:2] * v
    return {"basis": {"origin": p0.tolist(), "u": u.tolist(), "v": v.tolist()},
            "loops_uv": [outer_uv], "loops_3d": [outer_3d],
            "loop_areas": [float(abs(outer_area))]}


# =============================================================================
# 3-D figure style helpers
# =============================================================================

def _force_tick_size_3d(ax, size=None):
    """
    Matplotlib's 3D axes partially ignore tick_params for the Z axis and
    sometimes for X/Y too. This function directly sets the font size on
    every tick label Text object that already exists, guaranteeing the
    correct size regardless of rcParams or tick_params order issues.
    It ALSO re-asserts the label pad so the X/Y/Z titles stay clear of the
    cube after the ticks are finalised.
    Call this AFTER all data has been plotted (so ticks are finalised).
    """
    if size is None:
        size = FS_TICK
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        for label in axis.get_ticklabels():
            label.set_fontsize(size)
        axis.label.set_fontsize(FS_AXLABEL)
        axis.label.set_fontweight('bold')
        # NOTE: labelpad is intentionally NOT re-asserted here — doing so would
        # clobber the tighter per-call Z padding used for top-down panels.


def _style_ax3d(ax, title,
                xlabel="", ylabel="", zlabel="",
                elev=25, azim=-55, z_labelpad=None, z_tickpad=None):
    """
    Apply thesis-consistent style to a 3-D axes.

    Updated for a clean academic box:
      • axis labels pushed clear of the cube (AX3D_LABELPAD)
      • tick numbers padded off the grid lines (AX3D_TICKPAD_*)
      • a real white bounding box with crisp dark edges (panes filled)
      • on steep / top-down panels (elev >= AX3D_TOPDOWN_ELEV) the Z label and
        Z ticks use a tighter pad so they don't collide with the colorbar /
        Y axis. Override explicitly with z_labelpad / z_tickpad if needed.
    _force_tick_size_3d must be called AFTER drawing to guarantee tick sizes.
    """
    ax.set_facecolor('white')

    ax.set_title(title, fontsize=FS_SUBTITLE, fontweight='bold',
                 color='#111111', pad=AX3D_TITLE_PAD)

    # ── decide Z padding (auto-tighten for top-down unless overridden) ────────
    top_down = elev >= AX3D_TOPDOWN_ELEV
    if z_labelpad is None:
        z_labelpad = AX3D_Z_LABELPAD_TD if top_down else AX3D_LABELPAD
    if z_tickpad is None:
        z_tickpad = AX3D_Z_TICKPAD_TD if top_down else AX3D_TICKPAD_Z

    # ── axis labels: keep them horizontal and clear of the cube ───────────────
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_rotate_label(False)      # stop labels hugging the axis
        except Exception:
            pass
    ax.set_xlabel(xlabel, fontsize=FS_AXLABEL, fontweight='bold',
                  labelpad=AX3D_LABELPAD)
    ax.set_ylabel(ylabel, fontsize=FS_AXLABEL, fontweight='bold',
                  labelpad=AX3D_LABELPAD)
    ax.set_zlabel(zlabel, fontsize=FS_AXLABEL, fontweight='bold',
                  labelpad=z_labelpad)

    # ── tick numbers: push them off the grid lines ────────────────────────────
    ax.tick_params(axis='x', which='major', labelsize=FS_TICK,
                   pad=AX3D_TICKPAD_XY, width=1.4)
    ax.tick_params(axis='y', which='major', labelsize=FS_TICK,
                   pad=AX3D_TICKPAD_XY, width=1.4)
    ax.tick_params(axis='z', which='major', labelsize=FS_TICK,
                   pad=z_tickpad, width=1.4)

    # ── clean bounding box: white walls + crisp dark edges ────────────────────
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor(AX3D_PANE_FILL)
        pane.set_alpha(1.0)
        pane.set_edgecolor(AX3D_BOX_EDGE)
        pane.set_linewidth(AX3D_BOX_EDGE_LW)

    ax.grid(True, linestyle='--', alpha=0.25, color='#AAAAAA')
    ax.view_init(elev=elev, azim=azim)

    # thin out tick numbers so they stop bunching (esp. Z in the top-down view)
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=AX3D_MAX_TICKS))

    # shrink the cube inside the panel so tick numbers separate from the box
    # edges and labels get breathing room (version-robust across matplotlib).
    try:
        ax.set_box_aspect((1, 1, 0.85), zoom=0.85)   # matplotlib >= 3.6
    except TypeError:
        try:
            ax.set_box_aspect((1, 1, 0.85))          # matplotlib 3.3 - 3.5
        except Exception:
            pass
        try:
            ax.dist = 11                             # older fallback (bigger = smaller cube)
        except Exception:
            pass
    except Exception:
        pass


def _add_colorbar(fig, sm, ax_or_list, label):
    """
    Attach a colour bar with FS_CBAR label and FS_TICK tick labels.
    Both are set explicitly — rcParams alone is not enough for colorbars
    attached to 3-D axes.
    """
    cb = fig.colorbar(sm, ax=ax_or_list, shrink=0.55, pad=0.10, fraction=0.025)
    cb.set_label(label, fontsize=FS_CBAR, fontweight='bold', labelpad=12)
    cb.ax.tick_params(labelsize=FS_TICK, width=1.4, length=5, pad=4)
    # Force tick label size directly on the Text objects (belt-and-braces)
    for tl in cb.ax.get_yticklabels():
        tl.set_fontsize(FS_TICK)
    cb.outline.set_linewidth(0.8)
    return cb


def _fig3d_dual(suptitle, figsize=(26, 12)):
    """Figure with two 3-D subplots (isometric left + top-down right)."""
    fig = plt.figure(figsize=figsize, facecolor='white', dpi=DPI)
    fig.suptitle(suptitle, fontsize=FS_SUPTITLE, fontweight='bold',
                 color='#111111', y=0.97)
    ax_iso = fig.add_subplot(121, projection='3d')
    ax_top = fig.add_subplot(122, projection='3d')
    return fig, ax_iso, ax_top


def _save(fig, path):
    """
    Force tick sizes on every 3-D axes, then save WITHOUT bbox_inches='tight'.

    Older matplotlib does not include the 3-D Z-axis label in the tight-bbox
    calculation, so 'tight' crops 'Z (m)' off the right edge. Instead we reserve
    an explicit right-hand strip with subplots_adjust and save the full canvas.
    """
    for ax in fig.axes:
        if hasattr(ax, 'zaxis'):          # 3-D axis
            _force_tick_size_3d(ax)
    # reserve a right strip for the Z label; no tight crop (it eats the Z label)
    fig.subplots_adjust(left=0.03, right=0.90, top=0.90, bottom=0.05, wspace=0.18)
    fig.savefig(path, dpi=DPI, facecolor='white')
    plt.close(fig)
    info(f"Saved → {path}")


def _draw_noisy_3d(ax, noisy, step=8, alpha=0.20):
    ax.scatter(noisy[::step, 0], noisy[::step, 1], noisy[::step, 2],
               s=0.15, c='#BBBBBB', alpha=alpha, linewidths=0, depthshade=True)


def _expand_limits(ax, pts, pad_frac=AX3D_DATA_PAD):
    """
    Set X/Y/Z limits to the data extent PLUS a margin (pad_frac of the span)
    so geometry does not touch the bounding-box walls. Explicit set_*lim is
    used because matplotlib's 3-D axes ignore ax.margins() once limits are set.
    """
    if pts.shape[0] == 0: return
    for setter, getter, col in [(ax.set_xlim, ax.get_xlim, 0),
                                (ax.set_ylim, ax.get_ylim, 1),
                                (ax.set_zlim, ax.get_zlim, 2)]:
        lo, hi = float(pts[:, col].min()), float(pts[:, col].max())
        cur = getter()
        lo = min(lo, cur[0] if cur[0] != 0 else lo)
        hi = max(hi, cur[1] if cur[1] != 0 else hi)
        span = hi - lo
        pad  = span * pad_frac if span > 1e-9 else 1.0
        setter(lo - pad, hi + pad)


def _plane_patch_3d(lab, plane_dict, pts_all, labels):
    from scipy.spatial import ConvexHull
    pi  = plane_dict[lab]
    pts = pts_all[labels == lab]
    n   = pi['normal']; c = pi['centroid']
    ref = np.array([0., 0., 1.]) if abs(n[2]) < 0.9 else np.array([1., 0., 0.])
    uu  = np.cross(n, ref); uu /= np.linalg.norm(uu)
    vv  = np.cross(n, uu);  vv /= np.linalg.norm(vv)
    rel = pts - c
    uv  = np.column_stack([rel @ uu, rel @ vv])
    try:
        hull = ConvexHull(uv); h_uv = uv[hull.vertices]
    except Exception:
        h_uv = uv[np.random.choice(len(uv), min(4, len(uv)), replace=False)]
    return c + h_uv[:, 0:1] * uu + h_uv[:, 1:2] * vv


# =============================================================================
# 2-D (top-down XY) figure helper
# =============================================================================

def _fig2d(title, figsize=(12, 10)):
    """
    Create a single clean 2-D axes for top-down XY projections.
    All font sizes use the same FS_* constants as the 3-D figures.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', dpi=DPI)
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=FS_SUPTITLE, fontweight='bold',
                 color='#111111', pad=12)
    ax.set_xlabel('X (m)', fontsize=FS_AXLABEL, fontweight='bold',
                  color='#111111', labelpad=8)
    ax.set_ylabel('Y (m)', fontsize=FS_AXLABEL, fontweight='bold',
                  color='#111111', labelpad=8)
    ax.tick_params(axis='both', which='major', labelsize=FS_TICK,
                   width=1.6, length=6, colors='#111111', pad=5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35, color='#AAAAAA',
                  linewidth=0.9, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.35, color='#AAAAAA',
                  linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#666666')
        ax.spines[sp].set_linewidth(1.4)
    ax.set_aspect('equal')
    return fig, ax


def _add_colorbar_2d(fig, ax, sm, label):
    """Colourbar for a 2-D figure with consistent FS_* font sizes."""
    cb = fig.colorbar(sm, ax=ax, shrink=0.80, pad=0.02, fraction=0.030)
    cb.set_label(label, fontsize=FS_CBAR, fontweight='bold', labelpad=10)
    cb.ax.tick_params(labelsize=FS_TICK, width=1.4, length=5, pad=4)
    for tl in cb.ax.get_yticklabels():
        tl.set_fontsize(FS_TICK)
    cb.outline.set_linewidth(0.8)
    return cb


def _save_2d(fig, path):
    """Save a 2-D figure at DPI with tight layout."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    info(f"Saved → {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── 1. Load ───────────────────────────────────────────────────────────────
    info(f"Loading clusters : {NPY_CLUSTERS}")
    raw     = np.load(NPY_CLUSTERS)
    pts_all = raw[:, :3].astype(float)
    labels  = raw[:, 3].astype(int)
    info(f"Loading noisy    : {NPY_NOISY}")
    noisy   = np.load(NPY_NOISY)[:, :3].astype(float)
    info(f"Cluster pts={len(pts_all)}  Noisy pts={len(noisy)}")

    bb_all   = np.vstack([pts_all, noisy])
    bbox_min = np.percentile(bb_all, BBOX_ROBUST_PCT[0], axis=0) - BBOX_PAD
    bbox_max = np.percentile(bb_all, BBOX_ROBUST_PCT[1], axis=0) + BBOX_PAD

    # ── 2-5. Plane fit + normal snapping ──────────────────────────────────────
    info("Fitting planes and snapping normals ...")
    plane_dict = {}
    for lab in np.unique(labels):
        pts = pts_all[labels == lab]
        if len(pts) < 10:
            warn(f"  L{lab}: too few pts ({len(pts)}), skip"); continue

        c_raw, n_raw, _ = fit_plane_pca(pts)
        n_raw /= np.linalg.norm(n_raw)

        if abs(n_raw[2]) > HORIZ_NZ_THRESH:
            n_new  = np.array([0.0, 0.0, 1.0])
            z_mean = float(pts[:, 2].mean())
            pos    = np.array([c_raw[0], c_raw[1], z_mean])
            d_new  = -z_mean
            snap_type = "HORIZ"
        else:
            n_xy    = np.array([n_raw[0], n_raw[1], 0.0])
            xy_norm = float(np.linalg.norm(n_xy))
            if xy_norm < WALL_XY_NORM_MIN:
                warn(f"  L{lab}: near-zero XY norm ({xy_norm:.2e}), "
                     f"raw=[{n_raw[0]:+.4f},{n_raw[1]:+.4f},{n_raw[2]:+.4f}] — SKIPPED")
                continue
            n_new  = n_xy / xy_norm
            d_tmp  = -float(np.dot(n_new, c_raw))
            pos    = project_onto_plane(c_raw, n_new, d_tmp)
            d_new  = -float(np.dot(n_new, pos))
            snap_type = "WALL"

        is_horiz = abs(n_new[2]) > HORIZ_NZ_THRESH
        plane_dict[int(lab)] = {
            "normal":   n_new,  "d":       d_new,
            "centroid": pos,    "num_pts": len(pts),
            "is_horiz": is_horiz,
        }
        info(f"  L{lab:2d}: raw=[{n_raw[0]:+.3f},{n_raw[1]:+.3f},{n_raw[2]:+.3f}] → "
             f"snap=[{n_new[0]:+.3f},{n_new[1]:+.3f},{n_new[2]:+.3f}]  "
             f"pos=[{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]  "
             f"d={d_new:.3f}  {snap_type}")

    plane_labels = sorted(plane_dict)
    info(f"Plane dictionary: {len(plane_labels)} planes")
    plane_colors = plt.cm.tab20(np.linspace(0, 1, max(len(plane_labels), 1)))

    # =========================================================================
    # Fig 1 — Fitted planes: SINGLE isometric 3-D view
    # =========================================================================
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig1 = plt.figure(figsize=(16, 12), facecolor='white', dpi=DPI)
    fig1.suptitle(
        'Fitted Planes — Isometric View\n'
        '(coloured patches = plane extents, arrows = snapped normals)',
        fontsize=FS_SUPTITLE, fontweight='bold', color='#111111', y=0.97)

    ax1 = fig1.add_subplot(111, projection='3d')
    _style_ax3d(ax1, title='', xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                elev=25, azim=-55)

    for idx, lab in enumerate(plane_labels):
        pi  = plane_dict[lab]; col = plane_colors[idx]
        pts = pts_all[labels == lab]
        ax1.scatter(pts[::6, 0], pts[::6, 1], pts[::6, 2],
                    s=0.5, color=col, alpha=0.30, linewidths=0)
        try:
            pp    = _plane_patch_3d(lab, plane_dict, pts_all, labels)
            verts = [list(zip(pp[:, 0], pp[:, 1], pp[:, 2]))]
            patch = Poly3DCollection(verts, alpha=0.25,
                                     facecolor=col, edgecolor=col, linewidth=0.8)
            ax1.add_collection3d(patch)
        except Exception:
            pass
        c  = pi['centroid']; nv = pi['normal']
        ax1.quiver(c[0], c[1], c[2],
                   nv[0] * 2.5, nv[1] * 2.5, nv[2] * 2.5,
                   color=col, linewidth=2.0, arrow_length_ratio=0.25)
        ax1.text(c[0], c[1], c[2] + 0.5,
                 f"L{lab}{'(H)' if pi['is_horiz'] else ''}",
                 fontsize=FS_ANNOT, color=col, fontweight='bold', ha='center')

    # pad the cube around the plane points + normal arrow tips
    _expand_limits(ax1, pts_all[::4])
    _save(fig1, F_PLANES)

    # ── 6. Triple-plane vertices + intersection segments ──────────────────────
    info("Computing triple-plane vertices ...")
    raw_verts = []
    for i, j, k in combinations(plane_labels, 3):
        ns = [plane_dict[x]["normal"] for x in (i, j, k)]
        ds = [plane_dict[x]["d"]      for x in (i, j, k)]
        v  = lls_vertex(ns, ds, det_eps=DET_EPS)
        if v is None: continue
        if np.any(v < bbox_min) or np.any(v > bbox_max): continue
        if any(point_plane_abs(ns[t], ds[t], v) > PLANE_DIST_EPS for t in range(3)): continue
        raw_verts.append({"point": v, "planes": {i, j, k}})
    info(f"  Candidate vertices: {len(raw_verts)}")

    pair_to_pts = defaultdict(list)
    for rv in raw_verts:
        for a, b in combinations(list(rv["planes"]), 2):
            pair_to_pts[tuple(sorted((a, b)))].append(rv["point"])

    segments = []
    for i, j in combinations(plane_labels, 2):
        pair     = tuple(sorted((i, j)))
        pts_pair = pair_to_pts.get(pair, [])
        if len(pts_pair) < 2: continue
        pts_u = unique_pts(pts_pair)
        if pts_u.shape[0] < 2: continue
        dirv, dn = normalize(np.cross(plane_dict[i]["normal"], plane_dict[j]["normal"]))
        if dn < 1e-10: continue
        order = np.argsort(pts_u @ dirv)
        for a, b in zip(pts_u[order[:-1]], pts_u[order[1:]]):
            L = float(np.linalg.norm(b - a))
            if L < MIN_SEG_LEN: continue
            segments.append({"pair": pair, "a": a, "b": b, "length": L})
    info(f"  Intersection segments: {len(segments)}")

    all_verts_3d = (np.array([rv["point"] for rv in raw_verts])
                    if raw_verts else np.empty((0, 3)))

    # =========================================================================
    # Fig 2 — Candidate segments + triple-plane vertices (dual 3-D)
    # =========================================================================
    def _draw_fig2(ax, elev, azim, title):
        _style_ax3d(ax, title, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                    elev=elev, azim=azim)
        _draw_noisy_3d(ax, noisy)
        seg_lines = [np.array([s["a"], s["b"]]) for s in segments]
        if seg_lines:
            lc = Line3DCollection(seg_lines, colors='royalblue',
                                  linewidths=1.0, alpha=0.55)
            ax.add_collection3d(lc)
        if all_verts_3d.shape[0]:
            ax.scatter(all_verts_3d[:, 0], all_verts_3d[:, 1], all_verts_3d[:, 2],
                       s=25, c='red', zorder=5, depthshade=True,
                       label='Triple-plane vertex')
            leg = ax.legend(fontsize=FS_LEGEND, markerscale=2.0,
                            loc='upper left', framealpha=0.80,
                            edgecolor='#AAAAAA', fancybox=False)
            leg.get_frame().set_linewidth(0.8)
            for lh in leg.legend_handles:
                lh.set_sizes([60])
        all_pts = (np.vstack([noisy[::8], all_verts_3d])
                   if all_verts_3d.shape[0] else noisy[::8])
        _expand_limits(ax, all_pts)

    fig2, ax_iso2, ax_top2 = _fig3d_dual('Candidate Segments + Triple-Plane Vertices')
    _draw_fig2(ax_iso2, elev=25, azim=-55, title='Isometric view (elev=25°)')
    _draw_fig2(ax_top2, elev=70, azim=-60, title='Top-down view  (elev=70°)')
    _save(fig2, F_TRIPLE)

    # =========================================================================
    # Fig 2b — Candidate segments + triple-plane vertices (2-D top-down)
    # =========================================================================
    fig2b, ax2b = _fig2d('Candidate Segments + Triple-Plane Vertices')
    ax2b.scatter(noisy[::4, 0], noisy[::4, 1],
                 s=0.3, c='#BBBBBB', alpha=0.35, linewidths=0, zorder=1)
    for s in segments:
        ax2b.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]],
                  '-', color='royalblue', lw=0.9, alpha=0.55, zorder=2)
    if all_verts_3d.shape[0]:
        ax2b.scatter(all_verts_3d[:, 0], all_verts_3d[:, 1],
                     s=30, c='red', zorder=4, label='Triple-plane vertex')
        leg2b = ax2b.legend(fontsize=FS_LEGEND, markerscale=1.8,
                            loc='upper left', framealpha=0.80,
                            edgecolor='#AAAAAA', fancybox=False)
        leg2b.get_frame().set_linewidth(0.8)
    _save_2d(fig2b, F_TRIPLE_2D)


    # ── 7-8. Noisy-point density voting ───────────────────────────────────────
    info("Voting noisy points onto segments (fast KDTree) ...")
    from sklearn.neighbors import KDTree as _KDTree

    _rng      = np.random.default_rng(42)
    _N        = min(len(noisy), 500_000)
    _idx      = _rng.choice(len(noisy), _N, replace=False)
    noisy_sub = noisy[_idx]
    info(f"  Using {_N} / {len(noisy)} noisy pts for voting")

    seg_mids   = np.array([(s["a"] + s["b"]) / 2 for s in segments], float)
    _tree_segs = _KDTree(seg_mids)

    counts = np.zeros(len(segments), float)
    _ind   = _tree_segs.query_radius(noisy_sub, r=MAX_SNAP_DIST * 2)
    for pi, cands in enumerate(_ind):
        if len(cands) == 0: continue
        p = noisy_sub[pi]
        best_i, best_d = -1, 1e9
        for si in cands:
            d, _ = seg_pt_dist(p, segments[si]["a"], segments[si]["b"])
            if d < best_d: best_d = d; best_i = si
        if best_i >= 0 and best_d <= MAX_SNAP_DIST:
            counts[best_i] += 1.

    lengths      = np.array([s["length"] for s in segments], float)
    density      = counts / np.maximum(lengths, LENGTH_EPS)
    max_d        = float(density.max()) if density.size else 0.
    density_norm = density / max_d if max_d > 1e-12 else np.zeros_like(density)

    norm3 = plt.Normalize(density.min(),      density.max())
    norm4 = plt.Normalize(density_norm.min(), density_norm.max())
    sm3   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm3)
    sm4   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm4)
    rgba3 = np.array([sm3.to_rgba(density[si])      for si in range(len(segments))])
    rgba4 = np.array([sm4.to_rgba(density_norm[si]) for si in range(len(segments))])

    # =========================================================================
    # Fig 3 — All segments coloured by raw density (dual 3-D)
    # =========================================================================
    def _draw_fig3(ax, elev, azim, title):
        _style_ax3d(ax, title, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                    elev=elev, azim=azim)
        _draw_noisy_3d(ax, noisy)
        seg_lines = [np.array([s["a"], s["b"]]) for s in segments]
        if seg_lines:
            lc = Line3DCollection(seg_lines, linewidths=1.2, alpha=0.85)
            lc.set_color(rgba3)
            ax.add_collection3d(lc)
        _expand_limits(ax, noisy[::8])

    fig3, ax_iso3, ax_top3 = _fig3d_dual(
        'All Candidate Segments — Coloured by Linear Density')
    _draw_fig3(ax_iso3, elev=25, azim=-55, title='Isometric view (elev=25°)')
    _draw_fig3(ax_top3, elev=70, azim=-60, title='Top-down view  (elev=70°)')
    _add_colorbar(fig3, sm3, [ax_iso3, ax_top3], 'Linear density (votes / m)')
    _save(fig3, F_ALL_SEG)

    # =========================================================================
    # Fig 3b — All segments coloured by raw density (2-D top-down)
    # =========================================================================
    fig3b, ax3b = _fig2d('All Candidate Segments — Coloured by Linear Density')
    ax3b.scatter(noisy[::4, 0], noisy[::4, 1],
                 s=0.3, c='#BBBBBB', alpha=0.35, linewidths=0, zorder=1)
    for si, s in enumerate(segments):
        ax3b.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]],
                  '-', color=rgba3[si], lw=1.2, alpha=0.85, zorder=2)
    _add_colorbar_2d(fig3b, ax3b, sm3, 'Linear density (votes / m)')
    _save_2d(fig3b, F_ALL_SEG_2D)


    # =========================================================================
    # Fig 4 — Density heat-map with width-scaled lines (dual 3-D)
    # NOW SHOWS: (1) noisy points prominently, (2) cluster points coloured by
    #            group, (3) width-scaled segments coloured by normalised density
    # =========================================================================
    wn     = ((density_norm - density_norm.min()) /
              max(density_norm.max() - density_norm.min(), 1e-12))
    widths = 1.2 + wn * 6.0   # thicker range so width variation is more visible

    def _draw_fig4(ax, elev, azim, title):
        _style_ax3d(ax, title, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                    elev=elev, azim=azim)

        # ── (1) Noisy / boundary points — orange, more visible ───────────────
        ax.scatter(noisy[::6, 0], noisy[::6, 1], noisy[::6, 2],
                   s=0.8, c='#FF8C00', alpha=0.35, linewidths=0,
                   depthshade=True, label='Boundary / noisy pts')

        # ── (2) Cluster points coloured by group label ────────────────────────
        for idx, lab in enumerate(plane_labels):
            pts_lab = pts_all[labels == lab]
            col = plane_colors[idx]
            ax.scatter(pts_lab[::6, 0], pts_lab[::6, 1], pts_lab[::6, 2],
                       s=0.6, color=col, alpha=0.25, linewidths=0,
                       depthshade=True)

        # ── (3) Width-scaled density segments ────────────────────────────────
        for si, s in enumerate(segments):
            ax.plot([s["a"][0], s["b"][0]],
                    [s["a"][1], s["b"][1]],
                    [s["a"][2], s["b"][2]],
                    '-', color=rgba4[si], lw=float(widths[si]), alpha=0.92,
                    zorder=4)

        # Legend for noisy pts
        leg = ax.legend(fontsize=FS_LEGEND - 4, loc='upper left',
                        framealpha=0.80, edgecolor='#AAAAAA', fancybox=False)
        leg.get_frame().set_linewidth(0.8)
        _expand_limits(ax, noisy[::8])

    fig4, ax_iso4, ax_top4 = _fig3d_dual(
        'Linear Density Voting Heat-Map  (line width ∝ normalised density)')
    _draw_fig4(ax_iso4, elev=25, azim=-55, title='Isometric view (elev=25°)')
    _draw_fig4(ax_top4, elev=70, azim=-60, title='Top-down view  (elev=70°)')
    _add_colorbar(fig4, sm4, [ax_iso4, ax_top4], 'Normalised linear density')
    _save(fig4, F_HEATMAP)

    # =========================================================================
    # Fig 4b — Density heat-map with width-scaled lines (2-D top-down)
    # NOW SHOWS: (1) noisy points prominently, (2) cluster points coloured by
    #            group, (3) width-scaled segments coloured by normalised density
    # =========================================================================
    fig4b, ax4b = _fig2d('Linear Density Voting Heat-Map  (line width ∝ normalised density)')

    # ── (1) Cluster points coloured by group label ────────────────────────────
    for idx, lab in enumerate(plane_labels):
        pts_lab = pts_all[labels == lab]
        col = plane_colors[idx]
        ax4b.scatter(pts_lab[::4, 0], pts_lab[::4, 1],
                     s=0.8, color=col, alpha=0.22, linewidths=0,
                     zorder=1, label=f'L{lab}')

    # ── (2) Noisy / boundary points — orange, on top of cluster pts ──────────
    ax4b.scatter(noisy[::3, 0], noisy[::3, 1],
                 s=1.0, c='#FF8C00', alpha=0.40, linewidths=0,
                 zorder=2, label='Boundary / noisy pts')

    # ── (3) Width-scaled density segments ────────────────────────────────────
    for si, s in enumerate(segments):
        ax4b.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]],
                  '-', color=rgba4[si], lw=float(widths[si]),
                  alpha=0.92, zorder=3)

    # Legend — show noisy pts entry only (cluster legend would be too crowded)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8C00',
               markersize=8, label='Boundary / noisy pts'),
        Patch(facecolor='grey', alpha=0.4, label='Cluster pts (by group)'),
    ]
    ax4b.legend(handles=legend_handles, fontsize=FS_LEGEND - 4,
                loc='upper left', framealpha=0.80,
                edgecolor='#AAAAAA', fancybox=False)

    _add_colorbar_2d(fig4b, ax4b, sm4, 'Normalised linear density')
    _save_2d(fig4b, F_HEATMAP_2D)


    # ── 9. Filter segments ────────────────────────────────────────────────────
    if FILTER_MODE == "threshold":
        sel_idx = set(np.where(density_norm >= DENSITY_NORM_THRESH)[0].tolist())
        info(f"Filter norm_density>={DENSITY_NORM_THRESH} → "
             f"{len(sel_idx)}/{len(segments)}")
    else:
        k       = max(1, int(math.ceil(len(segments) * TOP_PCT / 100.)))
        sel_idx = set(np.argsort(density_norm)[::-1][:k].tolist())
        info(f"Filter top-{TOP_PCT}% → {len(sel_idx)}/{len(segments)}")

    is_sel   = np.array([i in sel_idx for i in range(len(segments))], bool)
    sel_segs = [si for si in range(len(segments)) if     is_sel[si]]
    non_segs = [si for si in range(len(segments)) if not is_sel[si]]

    norm5 = (plt.Normalize(density_norm[sel_segs].min(),
                            density_norm[sel_segs].max())
             if sel_segs else plt.Normalize(0, 1))
    sm5   = plt.cm.ScalarMappable(cmap=CMAP, norm=norm5)
    rgba5 = {si: sm5.to_rgba(density_norm[si]) for si in sel_segs}

    # =========================================================================
    # Fig 5 — Filtered segments (dual 3-D)
    # =========================================================================
    def _draw_fig5(ax, elev, azim, title):
        _style_ax3d(ax, title, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                    elev=elev, azim=azim)
        _draw_noisy_3d(ax, noisy)
        grey_lines = [np.array([segments[si]["a"], segments[si]["b"]])
                      for si in non_segs]
        if grey_lines:
            lc_g = Line3DCollection(grey_lines, colors='#CCCCCC',
                                    linewidths=0.5, alpha=0.35)
            ax.add_collection3d(lc_g)
        sel_lines = [np.array([segments[si]["a"], segments[si]["b"]])
                     for si in sel_segs]
        sel_rgba  = (np.array([rgba5[si] for si in sel_segs])
                     if sel_segs else np.empty((0, 4)))
        if sel_lines:
            lc_s = Line3DCollection(sel_lines, linewidths=2.2, alpha=0.95)
            lc_s.set_color(sel_rgba)
            ax.add_collection3d(lc_s)
        _expand_limits(ax, noisy[::8])

    fig5, ax_iso5, ax_top5 = _fig3d_dual('Filtered Segments After Density Threshold')
    _draw_fig5(ax_iso5, elev=25, azim=-55, title='Isometric view (elev=25°)')
    _draw_fig5(ax_top5, elev=70, azim=-60, title='Top-down view  (elev=70°)')
    _add_colorbar(fig5, sm5, [ax_iso5, ax_top5],
                  'Normalised density (selected segments)')
    _save(fig5, F_FILTERED)

    # =========================================================================
    # Fig 5b — Filtered segments (2-D top-down)
    # =========================================================================
    fig5b, ax5b = _fig2d('Filtered Segments After Density Threshold')
    ax5b.scatter(noisy[::4, 0], noisy[::4, 1],
                 s=0.3, c='#BBBBBB', alpha=0.35, linewidths=0, zorder=1)
    for si in non_segs:
        s = segments[si]
        ax5b.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]],
                  '-', color='#CCCCCC', lw=0.6, alpha=0.40, zorder=2)
    for si in sel_segs:
        s = segments[si]
        ax5b.plot([s["a"][0], s["b"][0]], [s["a"][1], s["b"][1]],
                  '-', color=rgba5[si], lw=2.2, alpha=0.95, zorder=3)
    _add_colorbar_2d(fig5b, ax5b, sm5, 'Normalised density (selected segments)')
    _save_2d(fig5b, F_FILTERED_2D)


    # ── 10-12. PSLG + CCW polygon ─────────────────────────────────────────────
    info("Tracing CCW polygons ...")
    plane_segs = defaultdict(list)
    for si in sel_segs:
        s = segments[si]
        for pid in s["pair"]:
            plane_segs[pid].append((s["a"], s["b"]))

    plane_polys = {}
    for lab in plane_labels:
        sl   = plane_segs.get(lab, [])
        if len(sl) < 3: continue
        poly = plane_polygon(plane_dict[lab], sl)
        if poly is None: continue
        plane_polys[lab] = poly
    info(f"Polygons traced: {len(plane_polys)}/{len(plane_labels)}")

    poly_colors_tab = plt.cm.tab20(np.linspace(0, 1, max(len(plane_polys), 1)))

    # =========================================================================
    # Fig 6 — Polygon formation
    # =========================================================================
    # (a) PyVista high-quality screenshot
    try:
        pl = pv.Plotter(off_screen=True, window_size=(2400, 1600))
        pl.set_background("white")
        for si in sel_segs:
            ln = pv.Line(segments[si]["a"], segments[si]["b"])
            pl.add_mesh(ln, color='#AAAAAA', opacity=0.25, line_width=1.)
        for idx, (lbl, poly) in enumerate(plane_polys.items()):
            nrm     = np.asarray(plane_dict[int(lbl)]["normal"], float)
            col_hex = '#%02x%02x%02x' % tuple(
                int(c * 255) for c in poly_colors_tab[idx][:3])
            for loop3d in poly["loops_3d"]:
                pts_off = np.asarray(loop3d, float) + 0.002 * nrm
                f = make_filled(pts_off)
                if f: pl.add_mesh(f, color=col_hex, opacity=0.30, show_edges=False)
                e = make_outline(pts_off)
                if e: pl.add_mesh(e, color=col_hex, opacity=1.0, line_width=5.)
        pl.camera_position = "iso"
        pv_path = F_POLYGON.replace(".png", "_pv.png")
        pl.show(screenshot=pv_path, auto_close=True, interactive=False)
        info(f"Saved → {pv_path}")
    except Exception as exc:
        warn(f"PyVista render failed: {exc}")

    # (b) Matplotlib dual 3-D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection as P3C

    def _draw_fig6(ax, elev, azim, title):
        _style_ax3d(ax, title, xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
                    elev=elev, azim=azim)
        _draw_noisy_3d(ax, noisy, step=10, alpha=0.15)
        sel_lines = [np.array([segments[si]["a"], segments[si]["b"]])
                     for si in sel_segs]
        if sel_lines:
            lc = Line3DCollection(sel_lines, colors='#CCCCCC',
                                  linewidths=0.6, alpha=0.30)
            ax.add_collection3d(lc)
        for idx, (lbl, poly) in enumerate(plane_polys.items()):
            col = poly_colors_tab[idx]
            for loop3d in poly["loops_3d"]:
                pts3  = np.asarray(loop3d, float)
                verts = [list(zip(pts3[:, 0], pts3[:, 1], pts3[:, 2]))]
                patch = P3C(verts, alpha=0.22, facecolor=col,
                            edgecolor=col, linewidth=2.0)
                ax.add_collection3d(patch)
                loop_c = np.vstack([pts3, pts3[0]])
                ax.plot(loop_c[:, 0], loop_c[:, 1], loop_c[:, 2],
                        '-', color=col, lw=2.5, alpha=1.0)
                cen = pts3.mean(0)
                ax.text(cen[0], cen[1], cen[2],
                        f"L{lbl}", fontsize=FS_ANNOT, color=col,
                        fontweight='bold', ha='center')
        _expand_limits(ax, noisy[::8])

    fig6, ax_iso6, ax_top6 = _fig3d_dual(
        'Polygon Formation — CCW Boundary Loops per Plane')
    _draw_fig6(ax_iso6, elev=25, azim=-55, title='Isometric view (elev=25°)')
    _draw_fig6(ax_top6, elev=70, azim=-60, title='Top-down view  (elev=70°)')
    _save(fig6, F_POLYGON)

    # ── 13. CSV ───────────────────────────────────────────────────────────────
    rows = []
    for si, s in enumerate(segments):
        rows.append({
            "seg_id":       si,
            "plane_i":      s["pair"][0],
            "plane_j":      s["pair"][1],
            "ax":           round(float(s["a"][0]), 4),
            "ay":           round(float(s["a"][1]), 4),
            "az":           round(float(s["a"][2]), 4),
            "bx":           round(float(s["b"][0]), 4),
            "by":           round(float(s["b"][1]), 4),
            "bz":           round(float(s["b"][2]), 4),
            "length":       round(float(lengths[si]), 4),
            "votes":        round(float(counts[si]),  2),
            "density":      round(float(density[si]), 4),
            "density_norm": round(float(density_norm[si]), 4),
            "selected":     bool(is_sel[si]),
        })
    with open(F_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    info(f"Saved → {F_CSV}")

    # ── 14. Geometry JSON ─────────────────────────────────────────────────────
    planes_out = []
    for lab in plane_labels:
        pi    = plane_dict[lab]
        entry = {"id":       int(lab),
                 "normal":   pi["normal"].tolist(),
                 "d":        float(pi["d"]),
                 "centroid": pi["centroid"].tolist(),
                 "is_horiz": bool(pi["is_horiz"]),
                 "num_pts":  pi["num_pts"]}
        if lab in plane_polys:
            poly = plane_polys[lab]
            entry["poly_loops_3d"]   = [lp.tolist() for lp in poly["loops_3d"]]
            entry["poly_loop_areas"] = [float(a)    for a  in poly["loop_areas"]]
        planes_out.append(entry)

    seg_out = [{"seg_id":       int(si),
                "pair":         list(segments[si]["pair"]),
                "a":            segments[si]["a"].tolist(),
                "b":            segments[si]["b"].tolist(),
                "length":       float(lengths[si]),
                "votes":        float(counts[si]),
                "density":      float(density[si]),
                "density_norm": float(density_norm[si])}
               for si in sel_segs]

    with open(F_GEO, "w") as f:
        json.dump({"config": {"HORIZ_NZ_THRESH":     HORIZ_NZ_THRESH,
                               "WALL_XY_NORM_MIN":    WALL_XY_NORM_MIN,
                               "DENSITY_NORM_THRESH": DENSITY_NORM_THRESH,
                               "MAX_SNAP_DIST":       MAX_SNAP_DIST,
                               "PLANE_DIST_EPS":      PLANE_DIST_EPS},
                   "planes":            planes_out,
                   "selected_segments": seg_out}, f, indent=2)
    info(f"Saved → {F_GEO}")

    # ── 15. PyRevit JSON ──────────────────────────────────────────────────────
    pyrevit = {"meta": {"num_planes": len(plane_polys),
                        "method": ("PCA-fit → normal-snap → "
                                   "triple-plane LLS → density vote → PSLG CCW")},
               "planes": []}
    for lbl, poly in plane_polys.items():
        if not poly["loops_3d"]: continue
        outer   = poly["loops_3d"][0]
        corners = [r3(pt) for pt in outer]
        if corners[0] != corners[-1]: corners.append(corners[0])
        pyrevit["planes"].append({
            "label":           int(lbl),
            "normal":          plane_dict[int(lbl)]["normal"].tolist(),
            "d":               round(float(plane_dict[int(lbl)]["d"]), 4),
            "is_horiz":        bool(plane_dict[int(lbl)]["is_horiz"]),
            "polygon_area_m2": round(float(poly["loop_areas"][0]), 3),
            "corners_ccw_3d":  corners,
        })
    with open(F_PYREVIT, "w") as f:
        json.dump(pyrevit, f, indent=2)
    info(f"Saved → {F_PYREVIT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    info("=" * 55)
    info(f"  Planes          : {len(plane_labels)}")
    info(f"  Triple vertices : {len(raw_verts)}")
    info(f"  Segments total  : {len(segments)}")
    info(f"  Segments kept   : {len(sel_segs)}")
    info(f"  Polygons traced : {len(plane_polys)}")
    info("[DONE]")


if __name__ == "__main__":
    main()