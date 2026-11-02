# -*- coding: utf-8 -*-
# ============================================================
# Envelope (walls/floors/roofs) + Windows + Doors placement
# Single unified pipeline
#
# Input JSON schemas supported (auto-detected per plane):
#   - "corners"        (open ring, legacy)
#   - "corners_ccw_3d" (closed CCW ring, PSLG output -> pyrevit_planes.json)
#
# Fixes integrated:
#   FIX 0 : Loader accepts BOTH "corners" and "corners_ccw_3d" and strips
#           the explicit closing-duplicate vertex.
#   FIX 1 : Safe FamilySymbol name access
#   FIX 2 : Safe NewFamilyInstance overload
#   FIX 3 : Opening center forced to match JSON center
#   FIX 4 : Sloped roof base elevation uses zmin instead of avg_z
#   FIX 5 : Better slope-edge selection for sloped roofs
#
#   NEW (dataset-robustness):
#   GUARD A : Duplicate / overlapping wall skip (over-segmented planes).
#   GUARD B : Footprint spike removal (near-reversal sliver vertices) for
#             floors and roofs.
#   REPORT  : Load-time dataset health validator. Prints which planes
#             overlap and which footprints have spikes BEFORE building,
#             so dirty datasets are visible even when the auto-guards
#             cannot fully repair them.
#
# NOTE: The guards are NO-OPS on clean datasets (simple boxes). They only
#       act when the specific defects are present, so they do not regress
#       datasets that already work. They are tolerance-based, not a
#       substitute for clean upstream segmentation. Tune the tolerances
#       (OVERLAP_TOL_M, SPIKE_MAX_TURN_DEG, WALL_COINCIDE_TOL_FT) per
#       project if needed.
# ============================================================

import json
import math
import clr

from pyrevit import revit

from Autodesk.Revit.DB import (
    XYZ,
    Line,
    Wall,
    WallType,
    Level,
    Floor,
    FloorType,
    RoofType,
    CurveLoop,
    CurveArray,
    ModelCurveArray,
    FilteredElementCollector,
    BuiltInParameter,
    AttachmentLocation,

    # openings
    FamilySymbol,
    BuiltInCategory,
    ElementTransformUtils,
)

from System.Collections.Generic import List
from System import Exception as SysException


# ----------------------------------------------------------------------
# 0. CONFIG
# ----------------------------------------------------------------------
#BIM_JSON = r"G:\CreateBIM\dataset\dataset1\polygon_results\bim_input_unified.json"
#BIM_JSON = r"G:\CreateBIM\dataset\dataset2\polygon_results\bim_input_unified.json"
#BIM_JSON = r"G:\CreateBIM\dataset\dataset3\polygon_results\bim_input_unified.json"
BIM_JSON = r"G:\CreateBIM\dataset\dataset4\polygon_results\bim_input_unified.json"
#BIM_JSON = r"G:\CreateBIM\dataset\dataset5\polygon_results\bim_input_unified.json"


# JSON coordinates are in meters
unit_scale = 3.28084  # meters -> feet for Revit XYZ

# Classification tolerances (still in meters, JSON space)
FLOOR_Z_TOL = 0.5
VERTICAL_MAX_ABS_NZ = 0.1
HORIZONTAL_MIN_ABS_NZ = 0.95

# Margins (feet)
MARGIN_FLOOR = 0.2 * unit_scale
MARGIN_ROOF  = 0.3 * unit_scale
ROOF_MARGIN_RETRY_FACTORS = [1.0, 0.2, 0.3, 0.6]  # [1.0, 0.6, 0.3, 0.0]

# Wall height behavior
WALL_TOP_EXTRA = 0.0 * unit_scale

# ---- Robustness guards (NEW) ----
# GUARD A: two vertical planes whose bottom centerlines are within this
#          distance AND are parallel are treated as the same wall.
WALL_COINCIDE_TOL_FT = 0.6          # feet (~18 cm)
WALL_PARALLEL_TOL_RAD = 0.10        # radians (~5.7 deg)
# GUARD B: footprint vertices whose boundary "turn" exceeds this are spikes.
SPIKE_MAX_TURN_DEG = 150.0
# REPORT: dataset health validator thresholds (meters / degrees, JSON space)
OVERLAP_TOL_M = 0.30
SPIKE_REPORT_TURN_DEG = 150.0

# Window family/type keyword filters (optional)
WINDOW_FAMILY_KEYWORD = None
WINDOW_TYPE_KEYWORD   = None

# Door family/type keyword filters (optional)
DOOR_FAMILY_KEYWORD = None
DOOR_TYPE_KEYWORD   = None

# Optional transforms for openings
APPLY_WINDOW_ROTATION = True
APPLY_WINDOW_FACING_FLIP = True

APPLY_DOOR_ROTATION = True
APPLY_DOOR_FACING_FLIP = True

doc = __revit__.ActiveUIDocument.Document
short_tol = doc.Application.ShortCurveTolerance  # feet


# ----------------------------------------------------------------------
# 1. Load unified JSON (planes + openings)
# ----------------------------------------------------------------------
def _strip_closing_duplicate(corners, eps=1e-9):
    """Remove an explicit closing vertex (first == last) from a ring."""
    if not corners or len(corners) < 2:
        return corners
    f = corners[0]
    l = corners[-1]
    if (abs(f[0] - l[0]) < eps and
            abs(f[1] - l[1]) < eps and
            abs(f[2] - l[2]) < eps):
        return corners[:-1]
    return corners


with open(BIM_JSON, "r") as f:
    data = json.load(f)

# ---- planes/envelope ----
raw_planes = []
if "planes" in data:
    raw_planes = data["planes"]
elif "envelopes" in data:
    raw_planes = data["envelopes"]

planes = []
for p in raw_planes:
    # Accept both schemas: legacy "corners" and PSLG "corners_ccw_3d".
    corners = p.get("corners_ccw_3d", p.get("corners", None))
    normal = p.get("normal", None)
    if corners is None or normal is None:
        continue

    corners = _strip_closing_duplicate(corners)

    planes.append({
        "label": int(p.get("label", p.get("id", p.get("plane_label", -1)))),
        "normal": normal,
        "corners": corners
    })

if not planes:
    raise SysException("No planes found in unified JSON: " + BIM_JSON)

# ---- openings ----
windows = data.get("windows", [])
doors   = data.get("doors", [])

# Backward compatibility: if unified JSON uses a single list
if (not windows) and (not doors) and ("openings" in data):
    for o in data.get("openings", []):
        t = (o.get("opening_type", o.get("type", "")) or "").lower()
        if t == "door":
            doors.append(o)
        else:
            windows.append(o)

print("[UNIFIED] planes:", len(planes), "| windows:", len(windows), "| doors:", len(doors))
print("[UNIFIED] source:", BIM_JSON)


# ----------------------------------------------------------------------
# 1b. DATASET HEALTH VALIDATOR (REPORT)  -- diagnostic only, no geometry
# ----------------------------------------------------------------------
def _bottom_line_xy_m(p, z_eps=0.1):
    """Two unique bottom-edge endpoints of a vertical plane (meters, XY)."""
    cs = p["corners"]
    if not cs:
        return None
    zmin = min(pt[2] for pt in cs)
    bot = [pt for pt in cs if abs(pt[2] - zmin) <= z_eps]
    uniq = []
    for pt in bot:
        if not any(abs(pt[0] - q[0]) < 1e-6 and abs(pt[1] - q[1]) < 1e-6 for q in uniq):
            uniq.append(pt)
    if len(uniq) < 2:
        return None
    return (uniq[0], uniq[1])

def validate_dataset(planes):
    print("\n===================== DATASET HEALTH REPORT =====================")
    verts = [p for p in planes if abs(p["normal"][2]) < VERTICAL_MAX_ABS_NZ]
    horiz = [p for p in planes if abs(p["normal"][2]) >= HORIZONTAL_MIN_ABS_NZ]

    # --- overlapping / duplicate vertical planes ---
    overlaps = []
    mids = {}
    for p in verts:
        e = _bottom_line_xy_m(p)
        if e is None:
            continue
        (x0, y0), (x1, y1) = (e[0][0], e[0][1]), (e[1][0], e[1][1])
        mids[p["label"]] = ((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                            math.atan2(y1 - y0, x1 - x0) % math.pi)
    labels = list(mids.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            ax, ay, aa = mids[labels[i]]
            bx, by, ba = mids[labels[j]]
            d = math.hypot(ax - bx, ay - by)
            dang = min(abs(aa - ba), math.pi - abs(aa - ba))
            if d < OVERLAP_TOL_M and dang < WALL_PARALLEL_TOL_RAD:
                overlaps.append((labels[i], labels[j], d))

    if overlaps:
        print("[WARN] Overlapping vertical planes (likely over-segmentation):")
        for a, b, d in overlaps:
            print("       planes {0} & {1}: centerlines {2:.3f} m apart -> 1 wall will be kept".format(a, b, d))
    else:
        print("[OK]   No overlapping vertical planes detected.")

    # --- footprint spikes in horizontal planes ---
    spike_hits = []
    for p in horiz:
        c = p["corners"]
        n = len(c)
        for i in range(n):
            a = c[(i - 1) % n]; b = c[i]; d = c[(i + 1) % n]
            v1x, v1y = b[0] - a[0], b[1] - a[1]
            v2x, v2y = d[0] - b[0], d[1] - b[1]
            l1 = math.hypot(v1x, v1y); l2 = math.hypot(v2x, v2y)
            if l1 < 1e-9 or l2 < 1e-9:
                spike_hits.append((p["label"], b[0], b[1], 180.0))
                continue
            cs = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (l1 * l2)))
            turn = math.degrees(math.acos(cs))
            if turn > SPIKE_REPORT_TURN_DEG:
                spike_hits.append((p["label"], b[0], b[1], turn))

    if spike_hits:
        print("[WARN] Footprint spikes / slivers (will be auto-removed before build):")
        for lab, x, y, t in spike_hits:
            print("       plane {0}: spike near ({1:.3f}, {2:.3f}) turn={3:.1f} deg".format(lab, x, y, t))
    else:
        print("[OK]   No footprint spikes detected in horizontal planes.")

    print("[INFO] planes={0} | vertical={1} | horizontal={2}".format(len(planes), len(verts), len(horiz)))
    print("=================================================================\n")
    return overlaps, spike_hits

validate_dataset(planes)


# ----------------------------------------------------------------------
# 2. Helpers: geometry + classification
# ----------------------------------------------------------------------
def plane_centroid_z_m(p):
    cs = p["corners"]
    if not cs:
        return 0.0
    return sum(pt[2] for pt in cs) / float(len(cs))

centroid_zs = [plane_centroid_z_m(p) for p in planes]
min_z_m = min(centroid_zs) if centroid_zs else 0.0

def classify_plane(p):
    n = p["normal"]
    nz = n[2]
    abs_nz = abs(nz)

    if abs_nz < VERTICAL_MAX_ABS_NZ:
        return "Wall"

    if abs_nz >= HORIZONTAL_MIN_ABS_NZ:
        zc = plane_centroid_z_m(p)
        return "Floor" if (zc <= min_z_m + FLOOR_Z_TOL) else "FlatRoof"

    return "SlopedRoof"

def get_corners_xyz_from_json(p):
    return [XYZ(pt[0] * unit_scale, pt[1] * unit_scale, pt[2] * unit_scale)
            for pt in p["corners"]]

def build_curveloop_from_ring(points):
    n = len(points)
    if n < 3:
        return None

    loop = CurveLoop()
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        if p1.DistanceTo(p2) < short_tol:
            continue
        loop.Append(Line.CreateBound(p1, p2))

    segs = [c for c in loop]
    if len(segs) < 3:
        return None
    return loop

def get_type_name(elem):
    p = elem.get_Parameter(BuiltInParameter.SYMBOL_NAME_PARAM)
    if p:
        return p.AsString() or ""
    return ""

def get_floor_type_by_keyword(keyword=None):
    coll = (FilteredElementCollector(doc)
            .OfClass(FloorType)
            .WhereElementIsElementType())
    types = list(coll)
    if not types:
        return None
    if keyword:
        kw = keyword.lower()
        for ft in types:
            if kw in get_type_name(ft).lower():
                return ft
    return types[0]

def get_roof_type_by_keyword(keyword=None):
    coll = (FilteredElementCollector(doc)
            .OfClass(RoofType)
            .WhereElementIsElementType())
    types = list(coll)
    if not types:
        return None
    if keyword:
        kw = keyword.lower()
        for rt in types:
            if kw in get_type_name(rt).lower():
                return rt
    return types[0]

def choose_slope_edges_for_plane(p, model_curves):
    """
    Improved slope-edge selection:
    - prefers edges perpendicular to the horizontal projection of the roof normal
    - among those, prefers the lowest edge
    - then prefers the longer edge
    """
    n = p["normal"]
    nx, ny = n[0], n[1]
    h_vec = XYZ(nx, ny, 0.0)
    h_len = h_vec.GetLength()
    if h_len < 1e-6:
        return []

    slope_dir = h_vec / h_len
    candidates = []

    for mc in model_curves:
        c = mc.GeometryCurve
        p0 = c.GetEndPoint(0)
        p1 = c.GetEndPoint(1)

        v = XYZ(p1.X - p0.X, p1.Y - p0.Y, 0.0)
        length = v.GetLength()
        if length < 1e-6:
            continue

        dir_xy = v / length
        perp_score = abs(dir_xy.DotProduct(slope_dir))
        edge_z = (p0.Z + p1.Z) / 2.0

        candidates.append((perp_score, edge_z, -length, mc))

    if not candidates:
        return []

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return [candidates[0][3]]

def attach_all_walls_to_roofs(doc, wall_ids, roof_ids):
    walls = [doc.GetElement(wid) for wid in wall_ids if doc.GetElement(wid) is not None]
    roofs = [doc.GetElement(rid) for rid in roof_ids if doc.GetElement(rid) is not None]

    print("[ATTACH] Newly created walls: {0}, roofs: {1}".format(len(walls), len(roofs)))

    if not walls or not roofs:
        print("[ATTACH] Nothing to attach for this run.")
        return

    n_ok = 0
    n_err = 0
    for w in walls:
        for r in roofs:
            try:
                w.AddAttachment(r.Id, AttachmentLocation.Top)
                n_ok += 1
            except Exception:
                n_err += 1

    print("[ATTACH] Successful wall-top attachments: {0}".format(n_ok))
    print("[ATTACH] Failed wall-top attempts: {0}".format(n_err))


# ----------------------------------------------------------------------
# 2a) GUARD A: duplicate / overlapping wall detection
# ----------------------------------------------------------------------
def wall_line_coincides(start, end, built_lines,
                        tol=WALL_COINCIDE_TOL_FT,
                        ang_tol=WALL_PARALLEL_TOL_RAD):
    """True if (start,end) nearly coincides (parallel + close midpoint)
    with any already-created wall centerline in built_lines."""
    mx = (start.X + end.X) / 2.0
    my = (start.Y + end.Y) / 2.0
    ang = math.atan2(end.Y - start.Y, end.X - start.X) % math.pi
    for (bx, by, bang) in built_lines:
        if math.hypot(mx - bx, my - by) < tol:
            dang = min(abs(ang - bang), math.pi - abs(ang - bang))
            if dang < ang_tol:
                return True
    return False

def register_wall_line(start, end, built_lines):
    built_lines.append((
        (start.X + end.X) / 2.0,
        (start.Y + end.Y) / 2.0,
        math.atan2(end.Y - start.Y, end.X - start.X) % math.pi
    ))


# ----------------------------------------------------------------------
# 2b) Polygon offset in XY
# ----------------------------------------------------------------------
def signed_area_xy(pts):
    a = 0.0
    n = len(pts)
    for i in range(n):
        p = pts[i]
        q = pts[(i + 1) % n]
        a += p.X * q.Y - q.X * p.Y
    return 0.5 * a

def intersect_lines_xy(p, r, q, s):
    den = r.X * s.Y - r.Y * s.X
    if abs(den) < 1e-9:
        return (False, 0.0)
    t = ((q.X - p.X) * s.Y - (q.Y - p.Y) * s.X) / den
    return (True, t)

def offset_polygon_xy(points, offset):
    n = len(points)
    if n < 3 or abs(offset) < 1e-9:
        return points

    ccw = signed_area_xy(points) > 0.0
    out = []

    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]

        e1 = XYZ(p1.X - p0.X, p1.Y - p0.Y, 0.0)
        e2 = XYZ(p2.X - p1.X, p2.Y - p1.Y, 0.0)

        l1 = e1.GetLength()
        l2 = e2.GetLength()
        if l1 < 1e-6 or l2 < 1e-6:
            out.append(p1)
            continue

        d1 = e1 / l1
        d2 = e2 / l2

        if ccw:
            n1 = XYZ(d1.Y, -d1.X, 0.0)
            n2 = XYZ(d2.Y, -d2.X, 0.0)
        else:
            n1 = XYZ(-d1.Y, d1.X, 0.0)
            n2 = XYZ(-d2.Y, d2.X, 0.0)

        aP = XYZ(p1.X + n1.X * offset, p1.Y + n1.Y * offset, 0.0)
        aD = d1
        bP = XYZ(p1.X + n2.X * offset, p1.Y + n2.Y * offset, 0.0)
        bD = d2

        ok, t = intersect_lines_xy(aP, aD, bP, bD)
        if not ok:
            navg = XYZ(n1.X + n2.X, n1.Y + n2.Y, 0.0)
            ln = navg.GetLength()
            if ln < 1e-6:
                out.append(p1)
            else:
                navg = navg / ln
                out.append(XYZ(p1.X + navg.X * offset, p1.Y + navg.Y * offset, p1.Z))
        else:
            x = aP.X + aD.X * t
            y = aP.Y + aD.Y * t
            out.append(XYZ(x, y, p1.Z))

    return out


# ----------------------------------------------------------------------
# 2c) Footprint validity checks + GUARD B (spike removal)
# ----------------------------------------------------------------------
def dedupe_ring_xy(points, eps=1e-5):
    if not points:
        return points

    out = [points[0]]
    for p in points[1:]:
        q = out[-1]
        if abs(p.X - q.X) > eps or abs(p.Y - q.Y) > eps:
            out.append(p)

    if len(out) > 2:
        p0 = out[0]
        pL = out[-1]
        if abs(p0.X - pL.X) <= eps and abs(p0.Y - pL.Y) <= eps:
            out.pop()

    return out

def remove_spikes_xy(points, max_turn_deg=SPIKE_MAX_TURN_DEG):
    """GUARD B: iteratively drop vertices where the boundary nearly reverses
    (a thin sliver/spike) or where an edge is zero-length. No-op on clean
    footprints. Operates on XYZ points (uses .X/.Y), preserves .Z of kept pts.
    """
    pts = list(points)
    changed = True
    while changed and len(pts) > 3:
        changed = False
        n = len(pts)
        for i in range(n):
            a = pts[(i - 1) % n]
            b = pts[i]
            c = pts[(i + 1) % n]
            v1x, v1y = b.X - a.X, b.Y - a.Y
            v2x, v2y = c.X - b.X, c.Y - b.Y
            l1 = math.hypot(v1x, v1y)
            l2 = math.hypot(v2x, v2y)
            if l1 < 1e-9 or l2 < 1e-9:
                pts.pop(i)
                changed = True
                break
            cs = (v1x * v2x + v1y * v2y) / (l1 * l2)
            cs = max(-1.0, min(1.0, cs))
            if math.degrees(math.acos(cs)) > max_turn_deg:
                pts.pop(i)
                changed = True
                break
    return pts

def segs_intersect_xy(a, b, c, d, eps=1e-9):
    def orient(p, q, r):
        return (q.X - p.X) * (r.Y - p.Y) - (q.Y - p.Y) * (r.X - p.X)

    def on_segment(p, q, r):
        return (min(p.X, r.X) - eps <= q.X <= max(p.X, r.X) + eps and
                min(p.Y, r.Y) - eps <= q.Y <= max(p.Y, r.Y) + eps)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    if abs(o1) <= eps and on_segment(a, c, b): return True
    if abs(o2) <= eps and on_segment(a, d, b): return True
    if abs(o3) <= eps and on_segment(c, a, d): return True
    if abs(o4) <= eps and on_segment(c, b, d): return True

    return False

def is_simple_polygon_xy(points, eps=1e-9):
    n = len(points)
    if n < 3:
        return False

    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]

        for j in range(i + 1, n):
            c = points[j]
            d = points[(j + 1) % n]

            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue

            shared = (
                (abs(a.X - c.X) < eps and abs(a.Y - c.Y) < eps) or
                (abs(a.X - d.X) < eps and abs(a.Y - d.Y) < eps) or
                (abs(b.X - c.X) < eps and abs(b.Y - c.Y) < eps) or
                (abs(b.X - d.X) < eps and abs(b.Y - d.Y) < eps)
            )
            if shared:
                continue

            if segs_intersect_xy(a, b, c, d, eps=eps):
                return False

    return True


# ----------------------------------------------------------------------
# Safe symbol selection
# ----------------------------------------------------------------------
def _safe_param_str(elem, bip):
    try:
        p = elem.get_Parameter(bip)
        if p:
            s = p.AsString()
            return s if s else ""
    except:
        pass
    return ""

def _symbol_family_name(sym):
    fam = _safe_param_str(sym, BuiltInParameter.SYMBOL_FAMILY_NAME_PARAM)
    if fam:
        return fam
    try:
        return sym.FamilyName or ""
    except:
        pass
    try:
        f = sym.Family
        return f.Name if f else ""
    except:
        return ""

def _symbol_type_name(sym):
    t = _safe_param_str(sym, BuiltInParameter.SYMBOL_NAME_PARAM)
    if t:
        return t
    try:
        return sym.Name or ""
    except:
        return ""

def get_first_symbol_by_category(doc, bic, fam_kw=None, type_kw=None):
    symbols = (FilteredElementCollector(doc)
               .OfCategory(bic)
               .OfClass(FamilySymbol)
               .WhereElementIsElementType()
               .ToElements())
    if not symbols:
        return None

    fam_kw_l = fam_kw.lower() if fam_kw else None
    typ_kw_l = type_kw.lower() if type_kw else None

    for s in symbols:
        famname = _symbol_family_name(s)
        typename = _symbol_type_name(s)

        if fam_kw_l and (fam_kw_l not in famname.lower()):
            continue
        if typ_kw_l and (typ_kw_l not in typename.lower()):
            continue
        return s

    return symbols[0]

def get_first_window_symbol(doc, fam_kw=None, type_kw=None):
    return get_first_symbol_by_category(doc, BuiltInCategory.OST_Windows, fam_kw, type_kw)

def get_first_door_symbol(doc, fam_kw=None, type_kw=None):
    return get_first_symbol_by_category(doc, BuiltInCategory.OST_Doors, fam_kw, type_kw)


# ----------------------------------------------------------------------
# Opening placement helpers
# ----------------------------------------------------------------------
def normalize_xyz(v):
    l = math.sqrt(v.X*v.X + v.Y*v.Y + v.Z*v.Z)
    if l < 1e-12:
        return v
    return XYZ(v.X/l, v.Y/l, v.Z/l)

def dot_xyz(a, b):
    return a.X*b.X + a.Y*b.Y + a.Z*b.Z

def project_point_to_plane(point_xyz, plane_p0_xyz, plane_n_xyz):
    n = normalize_xyz(plane_n_xyz)
    v = point_xyz - plane_p0_xyz
    dist = dot_xyz(v, n)
    return point_xyz - n.Multiply(dist)

def set_param_if_exists(elem, names, value_ft):
    for name in names:
        p = elem.LookupParameter(name)
        if p and (not p.IsReadOnly):
            try:
                p.Set(float(value_ft))
                return True
            except:
                pass
    return False

def get_instance_bbox_center(elem):
    bbox = elem.get_BoundingBox(None)
    if bbox is None:
        return None
    return XYZ(
        (bbox.Min.X + bbox.Max.X) / 2.0,
        (bbox.Min.Y + bbox.Max.Y) / 2.0,
        (bbox.Min.Z + bbox.Max.Z) / 2.0
    )

def get_instance_bbox_height(elem):
    bbox = elem.get_BoundingBox(None)
    if bbox is None:
        return None
    return bbox.Max.Z - bbox.Min.Z

def move_instance_center_to_target(doc, inst, target_center_xyz):
    doc.Regenerate()

    cur_center = get_instance_bbox_center(inst)
    if cur_center is None:
        return False

    delta = target_center_xyz - cur_center
    if delta.GetLength() < 1e-6:
        return True

    try:
        ElementTransformUtils.MoveElement(doc, inst.Id, delta)
        doc.Regenerate()
        return True
    except:
        return False

def best_wall_for_point(walls, target_xyz):
    best_wall = None
    best_d = None
    best_snap = None

    for w in walls:
        loc = w.Location
        if loc is None:
            continue
        crv = loc.Curve
        if crv is None:
            continue
        try:
            ir = crv.Project(target_xyz)
            if ir is None:
                continue
            p_on = ir.XYZPoint
            d = p_on.DistanceTo(XYZ(target_xyz.X, target_xyz.Y, p_on.Z))
        except:
            continue

        if best_d is None or d < best_d:
            best_wall = w
            best_d = d
            best_snap = p_on

    return best_wall, best_snap

def create_hosted_instance(doc, insert_pt, symbol, host_wall, level):
    try:
        return doc.Create.NewFamilyInstance(insert_pt, symbol, host_wall, level)
    except:
        from Autodesk.Revit.DB.Structure import StructuralType
        return doc.Create.NewFamilyInstance(insert_pt, symbol, host_wall, level, StructuralType.NonStructural)

def place_openings_from_list(openings, symbol, opening_kind,
                             walls_by_label, level,
                             apply_rotation=True,
                             apply_facing_flip=True):
    if symbol is None:
        print("[{0}] No FamilySymbol found. Load a {0} family first.".format(opening_kind.upper()))
        return

    print("[{0}] Selected family: {1} | type: {2}".format(
        opening_kind.upper(),
        _symbol_family_name(symbol),
        _symbol_type_name(symbol)
    ))

    if not symbol.IsActive:
        symbol.Activate()
        doc.Regenerate()

    placed = 0
    skipped = 0

    for o in openings:
        if o.get("status") != "assigned":
            continue

        label = o.get("host_plane_label", None)
        if label is None:
            skipped += 1
            continue

        wall_ids = walls_by_label.get(int(label), [])
        if not wall_ids:
            skipped += 1
            continue

        walls = [doc.GetElement(wid) for wid in wall_ids if doc.GetElement(wid) is not None]
        if not walls:
            skipped += 1
            continue

        try:
            oid  = o.get("id", "unknown")
            c_m  = o["center"]
            p0_m = o["host_plane_p0"]
            n_l  = o["host_plane_normal"]

            width_ft  = float(o["width"])  * unit_scale
            height_ft = float(o["height"]) * unit_scale
            angle = float(o.get("angle_rad", 0.0))

            center = XYZ(c_m[0]*unit_scale,  c_m[1]*unit_scale,  c_m[2]*unit_scale)
            p0     = XYZ(p0_m[0]*unit_scale, p0_m[1]*unit_scale, p0_m[2]*unit_scale)
            n      = XYZ(float(n_l[0]), float(n_l[1]), float(n_l[2]))
        except:
            skipped += 1
            continue

        target_center = project_point_to_plane(center, p0, n)

        host_wall, snap_on_line = best_wall_for_point(walls, target_center)
        if host_wall is None or snap_on_line is None:
            skipped += 1
            continue

        insert_pt = XYZ(snap_on_line.X, snap_on_line.Y, target_center.Z)

        try:
            inst = create_hosted_instance(doc, insert_pt, symbol, host_wall, level)
        except:
            skipped += 1
            continue

        w_ok = set_param_if_exists(inst, ["Width", "W", "WINDOW_WIDTH", "DOOR_WIDTH"], width_ft)
        h_ok = set_param_if_exists(inst, ["Height", "H", "WINDOW_HEIGHT", "DOOR_HEIGHT"], height_ft)

        doc.Regenerate()

        if apply_rotation and abs(angle) > 1e-6:
            axis = Line.CreateBound(insert_pt, insert_pt + XYZ(0, 0, 10))
            try:
                ElementTransformUtils.RotateElement(doc, inst.Id, axis, angle)
                doc.Regenerate()
            except:
                pass

        if apply_facing_flip:
            try:
                fo = inst.FacingOrientation
                if fo is not None and dot_xyz(fo, n) < 0.0:
                    inst.flipFacing()
                    doc.Regenerate()
            except:
                pass

        centered_ok = move_instance_center_to_target(doc, inst, target_center)

        actual_h = get_instance_bbox_height(inst)
        if actual_h is not None:
            sill_ft = (target_center.Z - actual_h / 2.0) - level.Elevation
            sill_p = inst.get_Parameter(BuiltInParameter.INSTANCE_SILL_HEIGHT_PARAM)
            if sill_p and (not sill_p.IsReadOnly):
                try:
                    sill_p.Set(sill_ft)
                    doc.Regenerate()
                    centered_ok = move_instance_center_to_target(doc, inst, target_center)
                except:
                    pass

        final_center = get_instance_bbox_center(inst)
        dz = None
        if final_center is not None:
            dz = final_center.Z - target_center.Z

        print("[{0}] {1} | resize W:{2} H:{3} | centered:{4} | final_dz_ft:{5}".format(
            opening_kind.upper(), oid, w_ok, h_ok, centered_ok, dz
        ))

        placed += 1

    print("[{0}] Placed: {1} | Skipped: {2}".format(opening_kind.upper(), placed, skipped))


# ----------------------------------------------------------------------
# 3. Default Level & Types
# ----------------------------------------------------------------------
level = (FilteredElementCollector(doc)
         .OfClass(Level)
         .FirstElement())
if level is None:
    raise SysException("No Level found in document.")

wall_type = (FilteredElementCollector(doc)
             .OfClass(WallType)
             .WhereElementIsElementType()
             .FirstElement())
if wall_type is None:
    raise SysException("No WallType found in document.")

floor_type_ground = get_floor_type_by_keyword("slab")
roof_type = get_roof_type_by_keyword("roof")

if floor_type_ground is None:
    raise SysException("No FloorType for ground slab (change 'slab' keyword).")
if roof_type is None:
    raise SysException("No RoofType found (change 'roof' keyword).")


# ----------------------------------------------------------------------
# 3b. Find global maximum roof Z
# ----------------------------------------------------------------------
roof_max_z_ft = None
for p in planes:
    kind = classify_plane(p)
    if kind in ["FlatRoof", "SlopedRoof"]:
        zs_m = [pt[2] for pt in p["corners"]]
        zmax_ft = max(zs_m) * unit_scale
        if roof_max_z_ft is None or zmax_ft > roof_max_z_ft:
            roof_max_z_ft = zmax_ft


# ----------------------------------------------------------------------
# 4. Transaction: Walls + Floors + Roofs + Windows + Doors
# ----------------------------------------------------------------------
with revit.Transaction("Create walls, floors, roofs + windows + doors"):

    created_wall_ids = []
    created_roof_ids = []
    walls_by_label = {}
    built_wall_lines = []     # GUARD A: centerlines already built
    skipped_dup_walls = 0     # GUARD A: count

    # ---------------- Walls ----------------
    for p in planes:
        if classify_plane(p) != "Wall":
            continue

        corners = get_corners_xyz_from_json(p)
        if len(corners) < 2:
            continue

        zs = [c.Z for c in corners]
        bottom_z = min(zs)

        if roof_max_z_ft is not None:
            target_top_z = roof_max_z_ft + WALL_TOP_EXTRA
            wall_height = target_top_z - bottom_z
        else:
            wall_height = (max(zs) - bottom_z)

        if wall_height < short_tol:
            continue

        z_eps = 0.10
        made_any = False

        for i in range(len(corners)):
            a = corners[i]
            b = corners[(i + 1) % len(corners)]

            if (abs(a.Z - bottom_z) <= z_eps) and (abs(b.Z - bottom_z) <= z_eps):
                start = XYZ(a.X, a.Y, level.Elevation)
                end = XYZ(b.X, b.Y, level.Elevation)

                if start.DistanceTo(end) < short_tol:
                    continue

                # GUARD A: skip a wall that coincides with one already built
                if wall_line_coincides(start, end, built_wall_lines):
                    skipped_dup_walls += 1
                    made_any = True   # treat as handled; don't fall to backup branch
                    continue

                line = Line.CreateBound(start, end)

                wall = Wall.Create(
                    doc,
                    line,
                    wall_type.Id,
                    level.Id,
                    wall_height,
                    bottom_z - level.Elevation,
                    False,
                    False,
                )
                created_wall_ids.append(wall.Id)
                walls_by_label.setdefault(p["label"], []).append(wall.Id)
                register_wall_line(start, end, built_wall_lines)
                made_any = True

        if not made_any:
            bottom_pts = [c for c in corners if abs(c.Z - bottom_z) <= z_eps]
            if len(bottom_pts) < 2:
                bottom_pts = sorted(corners, key=lambda pt: pt.Z)[:2]

            best_pair = None
            best_d2 = 0.0
            for i in range(len(bottom_pts)):
                for j in range(i + 1, len(bottom_pts)):
                    p0 = bottom_pts[i]
                    p1 = bottom_pts[j]
                    dx = p0.X - p1.X
                    dy = p0.Y - p1.Y
                    d2 = dx * dx + dy * dy
                    if d2 > best_d2:
                        best_d2 = d2
                        best_pair = (p0, p1)

            if best_pair:
                p0, p1 = best_pair
                start = XYZ(p0.X, p0.Y, level.Elevation)
                end = XYZ(p1.X, p1.Y, level.Elevation)

                if start.DistanceTo(end) >= short_tol:
                    # GUARD A also applies to the fallback branch
                    if wall_line_coincides(start, end, built_wall_lines):
                        skipped_dup_walls += 1
                    else:
                        line = Line.CreateBound(start, end)
                        wall = Wall.Create(
                            doc,
                            line,
                            wall_type.Id,
                            level.Id,
                            wall_height,
                            bottom_z - level.Elevation,
                            False,
                            False,
                        )
                        created_wall_ids.append(wall.Id)
                        walls_by_label.setdefault(p["label"], []).append(wall.Id)
                        register_wall_line(start, end, built_wall_lines)

    print("[WALLS] Created: {0} | Skipped as duplicates: {1}".format(
        len(created_wall_ids), skipped_dup_walls))

    # ---------------- Floors & Roofs ----------------
    for p in planes:
        kind = classify_plane(p)
        if kind not in ["Floor", "FlatRoof", "SlopedRoof"]:
            continue

        corners = get_corners_xyz_from_json(p)
        if len(corners) < 3:
            continue

        zs = [pt.Z for pt in corners]
        zmin = min(zs)
        zmax = max(zs)
        avg_z = sum(zs) / float(len(zs))

        # FIX: use lower elevation for sloped roof footprint
        if kind == "SlopedRoof":
            base_z = zmin
        else:
            base_z = avg_z

        offset = base_z - level.Elevation

        if kind == "Floor":
            ring = corners
            if MARGIN_FLOOR > 0.0:
                ring = offset_polygon_xy(ring, MARGIN_FLOOR)

            ring_flat = [XYZ(pt.X, pt.Y, avg_z) for pt in ring]
            ring_flat = dedupe_ring_xy(ring_flat, eps=1e-5)
            ring_flat = remove_spikes_xy(ring_flat)   # GUARD B

            if len(ring_flat) < 3:
                print("[WARN] Skipping Floor plane {0}: <3 unique footprint points.".format(p.get("label", -1)))
                continue

            if not is_simple_polygon_xy(ring_flat):
                if MARGIN_FLOOR > 0.0:
                    ring_flat_nom = [XYZ(pt.X, pt.Y, avg_z) for pt in corners]
                    ring_flat_nom = dedupe_ring_xy(ring_flat_nom, eps=1e-5)
                    ring_flat_nom = remove_spikes_xy(ring_flat_nom)   # GUARD B
                    if len(ring_flat_nom) >= 3 and is_simple_polygon_xy(ring_flat_nom):
                        ring_flat = ring_flat_nom
                        print("[FLOOR] Margin footprint invalid; fallback to no-margin for plane {0}.".format(p.get("label", -1)))
                    else:
                        print("[WARN] Skipping Floor plane {0}: footprint self-intersects.".format(p.get("label", -1)))
                        continue
                else:
                    print("[WARN] Skipping Floor plane {0}: footprint self-intersects.".format(p.get("label", -1)))
                    continue

            loop = build_curveloop_from_ring(ring_flat)
            if loop is None:
                print("[WARN] Skipping Floor plane {0}: CurveLoop could not be formed.".format(p.get("label", -1)))
                continue

            loops = List[CurveLoop]()
            loops.Add(loop)
            slab = Floor.Create(doc, loops, floor_type_ground.Id, level.Id)

            param = slab.get_Parameter(BuiltInParameter.FLOOR_HEIGHTABOVELEVEL_PARAM)
            if param and not param.IsReadOnly:
                param.Set(offset)

            continue

        made_roof = False
        base_corners = corners

        for factor in ROOF_MARGIN_RETRY_FACTORS:
            ring_try = base_corners
            m_try = MARGIN_ROOF * factor

            if m_try > 0.0:
                ring_try = offset_polygon_xy(ring_try, m_try)

            # FIX: flatten roof footprint to base_z, not avg_z
            ring_try_flat = [XYZ(pt.X, pt.Y, base_z) for pt in ring_try]
            ring_try_flat = dedupe_ring_xy(ring_try_flat, eps=1e-5)
            ring_try_flat = remove_spikes_xy(ring_try_flat)   # GUARD B

            if len(ring_try_flat) < 3:
                continue

            if not is_simple_polygon_xy(ring_try_flat):
                continue

            loop_try = build_curveloop_from_ring(ring_try_flat)
            if loop_try is None:
                continue

            curve_array = CurveArray()
            for seg in loop_try:
                p1_seg = seg.GetEndPoint(0)
                p2_seg = seg.GetEndPoint(1)
                curve_array.Append(Line.CreateBound(p1_seg, p2_seg))

            mca = ModelCurveArray()
            mca_ref = clr.Reference[ModelCurveArray](mca)

            try:
                roof = doc.Create.NewFootPrintRoof(curve_array, level, roof_type, mca_ref)
                created_roof_ids.append(roof.Id)

                param = roof.get_Parameter(BuiltInParameter.ROOF_LEVEL_OFFSET_PARAM)
                if param and not param.IsReadOnly:
                    param.Set(offset)

                if kind == "FlatRoof":
                    for mc in mca:
                        roof.set_DefinesSlope(mc, False)

                elif kind == "SlopedRoof":
                    nz = abs(p["normal"][2])
                    if nz > 1.0:
                        nz = 1.0
                    slope_angle = math.acos(nz)
                    slope_edges = choose_slope_edges_for_plane(p, mca)
                    for mc in mca:
                        if mc in slope_edges:
                            roof.set_DefinesSlope(mc, True)
                            roof.set_SlopeAngle(mc, slope_angle)
                        else:
                            roof.set_DefinesSlope(mc, False)

                if factor < 1.0:
                    print("[ROOF] Retry used factor={0:.2f} (margin_ft={1:.3f}) for plane {2}".format(
                        factor, m_try, p.get("label", -1)
                    ))

                made_roof = True
                break

            except Exception:
                continue

        if not made_roof:
            print("[WARN] Could not create roof for plane {0}".format(p.get("label", -1)))

    attach_all_walls_to_roofs(doc, created_wall_ids, created_roof_ids)

    # ---------------- Windows ----------------
    win_sym = get_first_window_symbol(doc, WINDOW_FAMILY_KEYWORD, WINDOW_TYPE_KEYWORD)
    place_openings_from_list(
        openings=windows,
        symbol=win_sym,
        opening_kind="window",
        walls_by_label=walls_by_label,
        level=level,
        apply_rotation=APPLY_WINDOW_ROTATION,
        apply_facing_flip=APPLY_WINDOW_FACING_FLIP
    )

    # ---------------- Doors ----------------
    door_sym = get_first_door_symbol(doc, DOOR_FAMILY_KEYWORD, DOOR_TYPE_KEYWORD)
    place_openings_from_list(
        openings=doors,
        symbol=door_sym,
        opening_kind="door",
        walls_by_label=walls_by_label,
        level=level,
        apply_rotation=APPLY_DOOR_ROTATION,
        apply_facing_flip=APPLY_DOOR_FACING_FLIP
    )