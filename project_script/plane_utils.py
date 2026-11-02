import numpy as np
import pandas as pd
import pyvista as pv

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
# Inputs (choose one pair)
# NPY_PATH  = r"H:\CreateBIM\dataset\Output\dbscan\d2k_1_subsampled_20cm_point_clusters.npy"
# NPY_NOISY = r"H:\CreateBIM\dataset\Output\dbscan\d2k_1_subsampled_20cm_noisy_points.npy"

# NPY_PATH  = r"H:\CreateBIM\dataset\Output\dbscan\d2k_1_original_point_clusters.npy"
# NPY_NOISY = r"H:\CreateBIM\dataset\Output\dbscan\d2k_1_original_noisy_points.npy"

# NPY_PATH  = r"H:\CreateBIM\dataset\Output\dbscan\d2k_2_original_point_clusters.npy"
# NPY_NOISY = r"H:\CreateBIM\dataset\Output\dbscan\d2k_2_original_noisy_points.npy"

# NPY_PATH  = r"H:\CreateBIM\dataset\Output\dbscan\d2k_5_point_clusters.npy"
# NPY_NOISY = r"H:\CreateBIM\dataset\Output\dbscan\d2k_5_noisy_points.npy"

# NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\d2k_1_original_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_1_original_noisy_points_transfer.npy"

# NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\d2k_2_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_2_noisy_points_transfer.npy"

# NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\d2k_3_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_3_noisy_points_transfer.npy"

#complex

# NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\complex_voxel_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\complex_voxel_noisy_points_transfer.npy"

# NPY_PATH  = r"F:\CreateBIM\dataset\d2k\complex\epsilon\complex_voxel_point_clusters_transfer_1_5.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\d2k\complex\epsilon\complex_voxel_noisy_points_transfer_1_5.npy"

#todo:Final_dataset complex

# NPY_PATH  = r"F:\CreateBIM\dataset\final_dataset\complex_final_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\final_dataset\complex_noisy_point_transfer.npy"

# NPY_PATH  = r"F:\CreateBIM\dataset\final_dataset\D1_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\final_dataset\D1_noisy_points_transfer.npy"

#todo: experiment

# NPY_PATH  = r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_points_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_noisy_points_transfer.npy"

NPY_PATH  = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_points_clusters_transfer.npy"
NPY_NOISY = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_noisy_points_transfer.npy"


data = np.load(NPY_PATH)
pts_all = data[:, :3].astype(float)
labels_all = data[:, 3].astype(int)

unique_labels = np.unique(labels_all)
print("Found labels:", unique_labels)

# (Optional) load noisy points if you want to visualize them later
try:
    noisy = np.load(NPY_NOISY)
    noisy_pts = noisy[:, :3].astype(float) if noisy.ndim == 2 and noisy.shape[1] >= 3 else None
except Exception:
    noisy_pts = None

# ----------------------------------------------------------------------
# 2. PCA plane fit helper
# ----------------------------------------------------------------------
def fit_plane_pca(points: np.ndarray):
    """
    Fit a plane using PCA (SVD).

    Returns:
        centroid (3,)
        normal   (3,) unit vector
        d        scalar in ax + by + cz + d = 0
    """
    centroid = points.mean(axis=0).astype(float)
    P0 = (points - centroid).astype(float)

    # SVD: smallest-variance axis is plane normal
    _, _, Vt = np.linalg.svd(P0, full_matrices=False)
    normal = Vt[-1].astype(float)

    nrm = np.linalg.norm(normal)
    if nrm == 0:
        # Degenerate case (all points identical)
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / nrm

    d = -float(np.dot(normal, centroid))
    return centroid, normal, d

# ----------------------------------------------------------------------
# 3. Fit and collect plane equations per cluster into a DataFrame
# ----------------------------------------------------------------------
plane_info = {}
rows = []

for lbl in unique_labels:
    pts = pts_all[labels_all == lbl]

    # Robust guard: need at least 3 points to define a plane
    if pts.shape[0] < 3:
        print(f"Skipping label {lbl}: not enough points ({pts.shape[0]})")
        continue

    centroid, normal, d = fit_plane_pca(pts)
    a, b, c = normal

    plane_info[int(lbl)] = {
        "centroid": centroid,
        "normal": normal,
        "d": d,
        "n_points": int(pts.shape[0]),
    }

    rows.append({
        "label": int(lbl),
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "n_points": int(pts.shape[0]),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "centroid_z": float(centroid[2]),
    })

planes_df = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
print(planes_df)

# ----------------------------------------------------------------------
# 4. Visualization with PyVista (arbitrary plane size)
# ----------------------------------------------------------------------
# One color per cluster (deterministic)
rng = np.random.RandomState(0)
cluster_colors = {int(lbl): rng.rand(3) for lbl in unique_labels}

# Default plane patch size
PLANE_SIZE = 15.0

# Optional per-label override sizes (must be numeric!)
PLANE_SIZES = {
    0: 25.0,  # IMPORTANT: label 0 exists in your output
    1: 25.0,
    2: 25.0,
    3: 15.0,
    4: 25.0,
    5: 15.0,
    6: 15.0,  # keep if you ever get label 6
}

plotter = pv.Plotter()
plotter.add_axes()
plotter.show_grid()

# Optional: show noisy points
if noisy_pts is not None and len(noisy_pts) > 0:
    plotter.add_points(
        pv.PolyData(noisy_pts),
        color="gray",
        point_size=2,
        render_points_as_spheres=True,
        opacity=0.25,
    )

for lbl in unique_labels:
    lbl = int(lbl)
    if lbl not in plane_info:
        continue  # skipped small clusters above

    pts = pts_all[labels_all == lbl]
    info = plane_info[lbl]

    centroid = np.asarray(info["centroid"], dtype=float)
    normal = np.asarray(info["normal"], dtype=float)
    d = float(info["d"])
    a, b, c = normal

    # Cluster points
    cloud = pv.PolyData(pts)
    plotter.add_points(
        cloud,
        color=cluster_colors[lbl],
        point_size=3,
        render_points_as_spheres=True,
        opacity=0.9,
    )

    # ✅ FIX: fallback must be numeric (not the dict)
    size = float(PLANE_SIZES.get(lbl, PLANE_SIZE))

    plane_mesh = pv.Plane(
        center=centroid,
        direction=normal,
        i_size=size,
        j_size=size,
        i_resolution=2,
        j_resolution=2,
    )

    plotter.add_mesh(
        plane_mesh,
        color=cluster_colors[lbl],
        opacity=0.9,
        show_edges=True,
    )

    # eq_text = f"L{lbl}: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0"
    # plotter.add_point_labels(
    #     [centroid],
    #     [eq_text],
    #     point_size=0,
    #     font_size=10,
    #     text_color="black",
    #     show_points=False,
    # )

plotter.show(title="PCA planes per cluster (arbitrary patch size)")
