import open3d as o3d
import numpy as np

def load_voxelized_as_numpy(path):
    """Load voxelized PLY and return Nx3 numpy array of XYZ points"""
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)   # shape (N, 3)
    print("Loaded voxelized dataset:", points.shape)
    return points


#----------------------------------------------------------------------------------------------------------
# Example usage
#--------------------------------------------------------------------------------------------------------------

#points_np = load_voxelized_as_numpy(r"G:\CreateBIM\dataset\dataset1\d2k_1_original_transfer.ply")

#points_np = load_voxelized_as_numpy(r"G:\CreateBIM\dataset\dataset2\d2k_2_transfer.ply")

#points_np = load_voxelized_as_numpy(r"G:\CreateBIM\dataset\dataset3\d2k_3_transfer.ply")

#points_np = load_voxelized_as_numpy(r"G:\CreateBIM\dataset\dataset4\experiment_1_voxel_transfer.ply")

points_np = load_voxelized_as_numpy(r"G:\CreateBIM\dataset\dataset5\experiment_2_transfer.ply")


# Pass to your model

# model.forward(points_np)
