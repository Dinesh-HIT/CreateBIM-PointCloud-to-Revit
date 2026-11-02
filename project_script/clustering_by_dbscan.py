import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
import building_model as bm
import PCA as pca
import os
os.environ["OMP_NUM_THREADS"] = "4"

# ------------------------------------------------------------------
# 1) Load
# ------------------------------------------------------------------
points = np.asarray(bm.points_np, dtype=np.float32)
print("[INFO] Total points:", points.shape[0])

# ------------------------------------------------------------------
# 2) Normal-based filtering (K-means on normal-deviation score)
# ------------------------------------------------------------------
normals, cos_sum, cos_var = pca.get_normals(points)

X = cos_sum.reshape(-1, 1)
km = KMeans(n_clusters=2, random_state=0).fit(X)

# choose cluster with smaller mean cos_sum as "good" (surface-like)
means = [X[km.labels_ == 0].mean(), X[km.labels_ == 1].mean()]
good = int(np.argmin(means))

keep_idx = np.where(km.labels_ == good)[0]
remove_idx = np.where(km.labels_ != good)[0]

points_filtered = points[keep_idx]
noisy_points = points[remove_idx]

print("[INFO] K-means filter kept:", points_filtered.shape[0],
      f"({points_filtered.shape[0]/points.shape[0]*100:.2f}%)",
      "| removed:", noisy_points.shape[0])
#------------------------------------------------------------------
# 3. quick visual- for filtering
#-----------------------------------------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_filtered)

colors = np.tile(np.array([[0.8, 0.8, 0.8]]), (points_filtered.shape[0], 1))
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="After Normal-based Filtering by K-means"
)
#------------------------------------------------------------------
# 3. quick visual- for noise
#-----------------------------------------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(noisy_points)

colors = np.tile(np.array([[0.8, 0.8, 0.8]]), (noisy_points.shape[0], 1))
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="noisy points after Normal-based Filtering by K-means"
)

# ------------------------------------------------------------------
# 3) DBSCAN only (on XYZ of filtered points)
# ------------------------------------------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_filtered)

eps = 1.1        # set from k-distance elbow with k=min_points
min_points = 6  # your chosen min_points (k in k-distance plot)

labels = np.asarray(
    pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True),
    dtype=int
)

# ---- cluster IDs excluding noise (-1) ----
cluster_ids = np.unique(labels[labels != -1])
n_clusters_excl_noise = cluster_ids.size
n_noise = int(np.sum(labels == -1))

print("[INFO] DBSCAN eps:", eps, "| min_points:", min_points)
print("[INFO] DBSCAN clusters (excluding noise):", n_clusters_excl_noise)
print("[INFO] DBSCAN cluster IDs (excluding noise):", cluster_ids)
print("[INFO] DBSCAN noise points:", n_noise)

# ------------------------------------------------------------------
# 4) Visualize (INLIERS ONLY, consistent with saved Nx4)
# ------------------------------------------------------------------
mask_inlier = labels != -1
points_inlier = points_filtered[mask_inlier]
labels_inlier = labels[mask_inlier].astype(np.int32)

if points_inlier.shape[0] == 0:
    print("[WARN] No inlier points after DBSCAN. Nothing to visualize/save.")
else:
    # Color map for inliers
    unique_inlier = np.unique(labels_inlier)
    colours = np.random.rand(unique_inlier.max() + 1, 3)

    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(points_inlier)
    pcd_final.colors = o3d.utility.Vector3dVector(colours[labels_inlier])

    o3d.visualization.draw_geometries(
        [pcd_final],
        window_name="DBSCAN only after K-means filtering (noise removed)"
    )

    # ------------------------------------------------------------------
    # 5) Save Nx4 (x,y,z,label) EXCLUDING DBSCAN noise (label = -1)
    # ------------------------------------------------------------------
    points_clusters = np.column_stack([points_inlier, labels_inlier])


#TODO: Point after normal deviation_based filtering and clusterin result are saved in NPY for future use

    # np.save(r"G:\CreateBIM\dataset\dataset1\dbscan_results\d2k_1_original_point_clusters_transfer.npy", points_clusters)
    # np.save(r"G:\CreateBIM\dataset\dataset1\dbscan_results\d2k_1_original_noisy_points_transfer.npy", noisy_points)

    # np.save(r"G:\CreateBIM\dataset\dataset2\dbscan_results\d2k_2_point_clusters_transfer.npy", points_clusters)
    # np.save(r"G:\CreateBIM\dataset\dataset2\dbscan_results\d2k_2_noisy_points_transfer.npy", noisy_points)

    # np.save(r"G:\CreateBIM\dataset\dataset3\dbscan_results\d2k_3_point_clusters_transfer.npy", points_clusters)
    # np.save(r"G:\CreateBIM\dataset\dataset3\dbscan_results\d2k_2_noisy_points_transfer.npy", points_clusters)


    # np.save(r"F:\CreateBIM\dataset\final_dataset\data_final_2_points_clusters_transfer.npy", points_clusters)
    # np.save(r"F:\CreateBIM\dataset\final_dataset\data_final_2_noisy_points_transfer.npy", noisy_points)
    #
    # np.save(r"G:\CreateBIM\dataset\dataset4\dbscan_results\experiment_1_voxel_points_clusters_transfer.npy", points_clusters)
    # np.save(r"G:\CreateBIM\dataset\dataset4\dbscan_results\experiment_1_voxel_noisy_points_transfer.npy", noisy_points)

    np.save(r"G:\CreateBIM\dataset\dataset5\dbscan_results\experiment_2_points_clusters_transfer.npy", points_clusters)
    np.save(r"G:\CreateBIM\dataset\dataset5\dbscan_results\experiment_2_noisy_points_transfer.npy", noisy_points)
