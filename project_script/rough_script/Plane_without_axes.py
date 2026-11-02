import os
import numpy as np
import pandas as pd
import pyvista as pv

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
# Select the dataset you want to visualize

# Dataset 1
# NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\d2k_1_original_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_1_original_noisy_points_transfer.npy"

# Dataset 2
# NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\d2k_2_point_clusters_transfer.npy"
# NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_2_noisy_points_transfer.npy"

# Dataset 3
NPY_PATH  = r"F:\CreateBIM\dataset\Output\dbscan\d2k_3_point_clusters_transfer.npy"
NPY_NOISY = r"F:\CreateBIM\dataset\Output\dbscan\d2k_3_noisy_points_transfer.npy"


# Output high-quality thesis figure
OUTPUT_PNG = r"F:\CreateBIM\dataset\Output\figures\dataset3_plane_fit_only.png"

# Create output folder if it does not exist
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)


data = np.load(NPY_PATH)
pts_all = data[:, :3].astype(float)
labels_all = data[:, 3].astype(int)

unique_labels = np.unique(labels_all)
print("Found labels:", unique_labels)


# ----------------------------------------------------------------------
# 2. PCA plane fit helper
# ----------------------------------------------------------------------
def fit_plane_pca(points: np.ndarray):
    """
    Fit a plane using PCA/SVD.

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
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / nrm

    d = -float(np.dot(normal, centroid))
    return centroid, normal, d


# ----------------------------------------------------------------------
# 3. Fit and collect plane equations per cluster
# ----------------------------------------------------------------------
plane_info = {}
rows = []

for lbl in unique_labels:
    lbl = int(lbl)
    pts = pts_all[labels_all == lbl]

    # Need at least 3 points to define a plane
    if pts.shape[0] < 3:
        print(f"Skipping label {lbl}: not enough points ({pts.shape[0]})")
        continue

    centroid, normal, d = fit_plane_pca(pts)
    a, b, c = normal

    plane_info[lbl] = {
        "centroid": centroid,
        "normal": normal,
        "d": d,
        "n_points": int(pts.shape[0]),
    }

    rows.append({
        "label": lbl,
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
print("\nPlane equations:")
print(planes_df)


# ----------------------------------------------------------------------
# 4. Visualization: fitted planes only, no point cloud, no axes, no grid
# ----------------------------------------------------------------------

# Use deterministic colors for clusters
rng = np.random.RandomState(0)
cluster_colors = {int(lbl): rng.rand(3) for lbl in unique_labels}

# Default plane size
PLANE_SIZE = 15.0

# Adjust these sizes according to each dataset/cluster if needed
PLANE_SIZES = {
    0: 25.0,
    1: 25.0,
    2: 25.0,
    3: 15.0,
    4: 25.0,
    5: 15.0,
    6: 15.0,
}

# High-resolution window for thesis-quality PNG
plotter = pv.Plotter(
    window_size=(2400, 1800),
    off_screen=True
)

# White background for thesis/report
plotter.set_background("white")

# Do NOT add axes or grid
# plotter.add_axes()
# plotter.show_grid()

for lbl in unique_labels:
    lbl = int(lbl)

    if lbl not in plane_info:
        continue

    info = plane_info[lbl]
    centroid = np.asarray(info["centroid"], dtype=float)
    normal = np.asarray(info["normal"], dtype=float)

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
        opacity=0.80,
        show_edges=True,
        edge_color="black",
        line_width=1.2,
        lighting=True,
        smooth_shading=True,
    )


# ----------------------------------------------------------------------
# 5. Freeze camera view
# ----------------------------------------------------------------------
# # Adjust only this part if the view angle is not ideal.
#
# plotter.camera_position = [
#     (20, -25, 18),   # camera position
#     (0, 0, 4),       # focal point
#     (0, 0, 1)        # view-up direction
# ]
#
# # Parallel projection gives cleaner technical/report figures
# plotter.enable_parallel_projection()
#
# # Optional zoom; increase/decrease if the model is too small/large
# plotter.camera.zoom(1.15)
#

# ----------------------------------------------------------------------
# 6. Save high-quality PNG
# ----------------------------------------------------------------------
plotter.screenshot(OUTPUT_PNG)
plotter.close()

print(f"\nHigh-quality plane-fit figure saved to:\n{OUTPUT_PNG}")