# CreateBIM — Point Cloud → BIM (Scan-to-BIM) Pipeline

This repository turns a raw **voxelized point cloud** of a building (`.ply`) into a
**parametric Revit model** (walls, floors, roofs, plus door and window openings).

The pipeline runs in four stages. Stages 1–3 are plain Python (CPython); stage 4
runs **inside Autodesk Revit via PyRevit** (IronPython). Each stage writes files
that the next stage reads, so the stages are loosely coupled and can be run/debugged
independently.

---

## Pipeline at a glance

```
  voxelized .ply
       │
       ▼
┌─────────────────────────────┐
│ building_model.py           │  load PLY → Nx3 numpy array (points_np)
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐   uses: PCA.py (normals + deviation score)
│ clustering_by_dbscan.py     │   1) KMeans normal-deviation filter (surface vs noise)
│  (Stage 1: filter + cluster)│   2) DBSCAN on clean points → planar clusters
└─────────────────────────────┘
       │   writes  *_points_clusters_transfer.npy  (Nx4: x,y,z,label)
       │   writes  *_noisy_points_transfer.npy     (Nx3)
       ▼
┌─────────────────────────────┐
│ polygon_reconstruction_     │   plane fit + normal snapping → plane-pair
│ optimized_normal.py         │   intersection segments → density vote/filter →
│  (Stage 2: planes→polygons) │   2D PSLG → CCW polygon loops → 3D
└─────────────────────────────┘
       │   writes  pyrevit_planes.json   (planes with corners_ccw_3d)
       │   writes  geometry_polygons.json, segments_density.csv, fig1..fig6 PNGs
       ▼
┌─────────────────────────────┐   + window/door point clouds (.ply)
│ Combine_envelope_opening.py │   cluster openings → find host wall plane →
│  (Stage 3: opening detect)  │   2D OBB (center/width/height) per opening
└─────────────────────────────┘
       │   writes  bim_input_unified.json  (envelope planes + opening records)
       ▼
┌─────────────────────────────┐   RUNS INSIDE REVIT (PyRevit / IronPython)
│ wall_door_window_uniified_  │   classify plane (wall/floor/roof) → CurveLoops →
│ JSON.py                     │   create Walls/Floors/Roofs → place door/window
│  (Stage 4: Revit build)     │   FamilyInstances at JSON centers/sizes
└─────────────────────────────┘
       │
       ▼
   Revit BIM model
```

---

## Files

| File | Stage | Runtime | Purpose |
|------|-------|---------|---------|
| `building_model.py` | — | CPython | Loads a voxelized `.ply` into `points_np` (Nx3). Single place to set the input dataset path. |
| `PCA.py` | helper | CPython | PCA, per-point normal estimation (KNN), and a normal-deviation score (`cos_sum`, `cos_var`) used to separate clean surfaces from noise. Also point-cloud visualization helpers. |
| `clustering_by_dbscan.py` | 1 | CPython | KMeans normal-deviation filter + Open3D DBSCAN clustering. Saves cluster and noise `.npy`. |
| `polygon_reconstruction_optimized_normal.py` | 2 | CPython | Fits planes, reconstructs wall/floor/roof polygons, writes `pyrevit_planes.json` and thesis-quality figures. Standalone script with a `main()`. |
| `Combine_envelope_opening.py` | 3 | CPython | Detects window/door openings, assigns each to a host wall, and merges everything into `bim_input_unified.json`. |
| `wall_door_window_uniified_JSON.py` | 4 | **Revit + PyRevit** | Reads the unified JSON and builds the Revit model (walls, floors, roofs, openings). |

---

## Requirements

**Stages 1–3 (CPython, e.g. Python 3.9+):**

```bash
pip install numpy open3d scikit-learn pandas pyntcloud matplotlib pyvista
```

- `numpy`, `open3d` — arrays, PLY IO, DBSCAN, visualization
- `scikit-learn` — KMeans (Stage 1) and DBSCAN (Stage 3)
- `pandas`, `pyntcloud` — used by `PCA.py` for KNN normal estimation
- `matplotlib`, `pyvista` — figure generation in Stage 2

**Stage 4 (Revit):**

- Autodesk Revit + the **PyRevit** add-in (runs on IronPython, not CPython).
- Uses the Revit API (`Autodesk.Revit.DB`) and expects suitable **wall / floor / roof
  types** and **door / window families** to be loaded in the active Revit document.
- This script **cannot be run from the normal Python interpreter** — open the project
  in Revit and run it through PyRevit.

---

## How to run

> All scripts currently use **hard-coded Windows paths** (e.g. `G:\CreateBIM\dataset\...`).
> Before running, edit the path/parameter lines at the top of each script (or in `main()`)
> to point at your own data. Datasets 1–5 are switched by commenting/uncommenting blocks.

### Stage 0 — set the input cloud
In `building_model.py`, point `load_voxelized_as_numpy(...)` at your voxelized `.ply`.

### Stage 1 — filter + cluster
```bash
python clustering_by_dbscan.py
```
Imports `building_model` and `PCA`. Opens Open3D windows showing (a) filtered points,
(b) removed noise, (c) DBSCAN clusters. On close it saves:
- `<dataset>/dbscan_results/<name>_points_clusters_transfer.npy` — `Nx4` `(x, y, z, label)`, noise excluded
- `<dataset>/dbscan_results/<name>_noisy_points_transfer.npy` — `Nx3` removed/noise points

Tune `eps` and `min_points` (k-distance elbow) near the top of the DBSCAN section.

### Stage 2 — polygon reconstruction
Set `NPY_CLUSTERS`, `NPY_NOISY`, and `OUT_DIR` at the top of
`polygon_reconstruction_optimized_normal.py`, then:
```bash
python polygon_reconstruction_optimized_normal.py
```
Writes into `OUT_DIR`:
- `pyrevit_planes.json` — per-plane normal, offset, and `corners_ccw_3d` polygon (the key Revit input)
- `geometry_polygons.json`, `segments_density.csv`
- `fig1_fitted_planes_3d.png` … `fig6_polygons_3d.png` (+ `*_2d.png` variants), 300 DPI

### Stage 3 — openings + unified JSON
Edit `main()` of `Combine_envelope_opening.py` (`planes_json`, `windows_ply`,
`doors_ply`, `out_path`) then:
```bash
python Combine_envelope_opening.py
```
Reads `pyrevit_planes.json` plus the **window/door point clouds**, clusters the
openings, assigns a host wall to each, and writes `bim_input_unified.json`
(envelope planes + opening records). See the schema below.

### Stage 4 — build in Revit
Open the target project in Revit, set `BIM_JSON` at the top of
`wall_door_window_uniified_JSON.py` to your `bim_input_unified.json`
(or `pyrevit_planes.json` for envelope-only), and run it via PyRevit. It classifies
each plane as wall/floor/roof, builds the geometry, attaches walls to roofs, and places
door/window family instances at the JSON-specified centers and sizes.

---

## Key parameters to tune

| Stage | Parameter | Default | Meaning |
|-------|-----------|---------|---------|
| 1 | `KMeans(n_clusters=2)` on `cos_sum` | — | Splits "surface-like" (low normal deviation) from "noisy" points. |
| 1 | `eps` | `1.1` | DBSCAN neighborhood radius (point-cloud units). |
| 1 | `min_points` | `6` | DBSCAN min cluster density (k in the k-distance plot). |
| 2 | `HORIZ_NZ_THRESH` / `VERT_NZ_THRESH` | `0.90 / 0.10` | Normal-snapping cutoffs: `|n_z|>0.90`→horizontal, `<0.10`→vertical wall, else sloped. |
| 2 | density threshold | — | Segments are kept by votes-per-length from noisy points; below threshold are dropped. |
| 3 | `window_eps`, `window_min_samples` | `0.50, 10` | DBSCAN for window clusters. |
| 3 | `door_eps`, `door_min_samples` | `0.30, 25` | DBSCAN for door clusters. |
| 3 | `wall_nz_threshold` | `0.5` | A plane counts as a wall (opening host) if `|n_z|` < this. |
| 3 | `tol_plane` | `0.12 m` | Strict max distance from opening centroid to host wall. |
| 3 | `max_plane_fallback` | `2.0 m` | If no wall within `tol_plane`, accept nearest wall whose polygon still contains the centroid, up to this distance. Set = `tol_plane` to disable. |
| 4 | `OVERLAP_TOL_M`, `SPIKE_MAX_TURN_DEG`, `WALL_COINCIDE_TOL_FT` | see file | Robustness guards: duplicate-wall skip and footprint spike removal. |

---

## `bim_input_unified.json` — opening record schema

Each detected window/door is stored with its host wall and a 2-D oriented bounding box:

```json
{
  "center":            [x, y, z],
  "width":             0.0,
  "height":            0.0,
  "angle_rad":         0.0,
  "sill_v_plane":      0.0,
  "units":             "m",
  "host_plane_label":  0,
  "host_plane_normal": [nx, ny, nz],
  "host_plane_u":      [ux, uy, uz],
  "host_plane_v":      [vx, vy, vz],
  "host_plane_p0":     [px, py, pz]
}
```

Coordinates are in **meters**; Stage 4 converts to feet internally for the Revit API.

---

## Notes & known limitations

- **Hard-coded paths.** Every script targets `G:\CreateBIM\dataset\...`. Update these
  to your environment before running; this is the most common cause of "file not found".
- **Stage 4 needs Revit.** `wall_door_window_uniified_JSON.py` imports `clr`, `pyrevit`,
  and `Autodesk.Revit.DB`; it only runs inside Revit via PyRevit, never in plain CPython.
  Required wall/floor/roof types and door/window families must already exist in the document.
- **Normal estimation is heavy.** `PCA.get_normals` uses `knn=1000` neighbors per point,
  which is accurate but slow on large clouds — reduce `knn` if performance matters.
- **Robustness guards are no-ops on clean data.** Stage 4's duplicate-wall and
  spike-removal guards only act when those defects are present; they are tolerance-based
  and not a substitute for clean upstream segmentation. Tune tolerances per project.
- **Door pipeline.** In `Combine_envelope_opening.py`, the door branch in `main()` is
  currently commented out (windows only). Uncomment it and set `doors_ply` to also export doors.
- **Filename typo.** `wall_door_window_uniified_JSON.py` ("uniified") keeps the original
  spelling; update imports/paths consistently if you rename it.
- **Some commented save lines mismatch.** A few commented-out `np.save(... noisy ...)`
  lines in `clustering_by_dbscan.py` pass `points_clusters` instead of `noisy_points`;
  verify the variable before uncommenting.

---

## Glossary

- **Normal-deviation score (`cos_sum`)** — sum of angles between a point's normal and its
  neighbors'. High score = inconsistent normals = edge/noise; low = flat surface.
- **Normal snapping** — forcing a fitted plane normal to a clean horizontal, vertical, or
  sloped orientation so reconstructed walls/floors are axis-consistent.
- **PSLG** — Planar Straight-Line Graph; the 2-D vertex/edge graph traversed CCW to form
  closed polygon loops.
- **OBB** — Oriented Bounding Box; the rotated rectangle fit to an opening's points to get
  its center, width, height, and angle.
