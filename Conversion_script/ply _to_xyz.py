import open3d as o3d
import numpy as np

# --------- CONFIGURATION ---------
#PLY_PATH =r"H:\CreateBIM\dataset\d2k\d2k_1_subsampled_20cm.ply"
#PLY_PATH =r"H:\CreateBIM\dataset\d2k\d2k_2_original.ply"
#PLY_PATH =r"H:\CreateBIM\dataset\d2k\d2k_5.ply"
#PLY_PATH =r"H:\CreateBIM\dataset\d2k\d2k_1_window_points.ply"
#PLY_PATH =r"H:\CreateBIM\dataset\d2k\d2k_1_door_points.ply"
#PLY_PATH =r"F:\CreateBIM\dataset\d2k\d2k_1_original_transfer.ply"
#PLY_PATH =r"F:\CreateBIM\dataset\d2k\d2k_3_transfer.ply"
#PLY_PATH =r"F:\CreateBIM\dataset\d2k\d2k_2_transfer.ply"
#PLY_PATH = r"F:\CreateBIM\dataset\d2k\d2k_1_original_window_transfer.ply"
#PLY_PATH = r"F:\CreateBIM\dataset\d2k\d2k_1_original_door_transfer.ply"
#PLY_PATH = r"F:\CreateBIM\dataset\d2k\d2k_1_original_window_transfer.ply"
#PLY_PATH = r"F:\CreateBIM\dataset\final_dataset\data_final_1_transfer.ply"
PLY_PATH = r"G:\CreateBIM\dataset\dataset4\experiment_1_voxel_transfer.ply"

#PLY_PATH = r"G:\CreateBIM\dataset\dataset2\d2k_2_window_3_transfer.ply"
#PLY_PATH = r"G:\CreateBIM\dataset\dataset3\d2k_3_window_transfer_again.ply"
#PLY_PATH = r"G:\CreateBIM\dataset\dataset4\experiment_1_window_transfer.ply"

#todo:Output

#OUT_XYZ_PATH = r"F:\CreateBIM\dataset\final_dataset\XYZ\final_data_1_transfer.xyz"


OUT_XYZ_PATH = r"G:\CreateBIM\dataset\dataset4\XYZ\experiment_1_voxel_transfer.xyz"

N_SAMPLE_POINTS = 500_000  # only used if it's a real mesh
# ---------------------------------

print(f"Loading PLY: {PLY_PATH}")

# 1) Try to load as a point cloud first
pc = o3d.io.read_point_cloud(PLY_PATH)

if not pc.is_empty():
    print("Loaded as PointCloud.")
else:
    print("Failed to load as PointCloud, trying TriangleMesh...")
    mesh = o3d.io.read_triangle_mesh(PLY_PATH)
    if mesh.is_empty():
        raise ValueError("Could not load PLY as point cloud or mesh.")
    if len(mesh.triangles) == 0:
        raise ValueError("Mesh has no triangles (only vertices). Cannot sample points.")
    print("Loaded as TriangleMesh, sampling points on surface...")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    pc = mesh.sample_points_poisson_disk(N_SAMPLE_POINTS)

# Now pc should be a valid point cloud
points = np.asarray(pc.points)
if points.shape[0] == 0:
    raise ValueError("Point cloud has no points.")

print(f"Number of points to export: {points.shape[0]}")

# Save XYZ (x y z)
np.savetxt(OUT_XYZ_PATH, points, fmt="%.6f")
print(f"Saved XYZ point cloud to: {OUT_XYZ_PATH}")
print("You can now import this .xyz into Autodesk ReCap to create an .rcp file.")
