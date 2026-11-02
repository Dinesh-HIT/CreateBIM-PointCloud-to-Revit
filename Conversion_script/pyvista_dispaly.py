import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

# 1) Load your (N,4) array: [x, y, z, cluster_id]
xyz_labels = np.load(r"H:\CreateBIM\data\building3_vox_10_4C_xyz_labels_twoSides.npy")

points = xyz_labels[:, :3]             # (N,3)
labels = xyz_labels[:, 3].astype(int)  # (N,)

print("Points shape:", points.shape)
print("Unique cluster IDs:", np.unique(labels))

# 2) Build a color for each cluster ID (using a stable colormap)
unique_ids = np.unique(labels)
cmap = plt.get_cmap("tab20")   # 20 distinct colors, repeat if more clusters

id_to_color = {}
for i, cid in enumerate(unique_ids):
    rgba = cmap(i % cmap.N)    # RGBA in [0,1]
    id_to_color[cid] = rgba[:3]

# Create an (N,3) array of RGB colors for each point
cluster_colors = np.vstack([id_to_color[cid] for cid in labels])

# 3) Create PyVista cloud and attach colors
cloud = pv.PolyData(points)
cloud["rgb"] = cluster_colors   # values in [0,1]

# 4) Visualize
plotter = pv.Plotter()
plotter.add_points(cloud, scalars="rgb", rgb=True, point_size=3)
plotter.add_axes()
plotter.show()
