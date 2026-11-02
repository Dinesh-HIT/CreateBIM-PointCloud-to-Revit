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
