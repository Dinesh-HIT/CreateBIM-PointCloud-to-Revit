import open3d as o3d

#path = r"G:\pointcloud\Best_building\Building_20cm_best.ply"
path = r"G:\pointcloud\divyantika_building\divyantika_20cm.ply"

# 读取点云文件
pcd = o3d.io.read_point_cloud(path)

# 打印点云信息并显示
print(pcd)
o3d.visualization.draw_geometries([pcd])