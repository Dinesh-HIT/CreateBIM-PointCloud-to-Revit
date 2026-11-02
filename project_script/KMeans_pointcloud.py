import os
os.environ["OMP_NUM_THREADS"] = '4'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import building_model as bm
import PCA as pca
import Rotation_and_translate as rnt

import pyvista as pv


# 数据导入
# 载入模型
model = bm.points_np
points = model
rotation = False
noisy = False

# 添加随机噪声
if noisy:
    noise = np.random.uniform(low=-0.5, high=0.5, size=points.shape)
    points = points + noise

# 模型平移旋转和缩放
if rotation:
    alfa = -30  # 绕x轴旋转的角度
    beta = 19  # 绕y轴旋转的角度
    gama = 47  # 绕z轴旋转的角度
    move = [0, 20, 20]  # 旋转后平移的坐标大小
    points = rnt.transform_points(points, alfa, beta, gama, move[0], move[1], move[2])
    # 缩放模型
    points = points/2
    # 规范模型于立方体世界空间
    world_space_size = 60  # 立方体世界空间的边长
    # points = rnt.filter_points(points, world_space_size)

# 计算点云法向量
normals, cos_sum, cos_var = pca.get_normals(points)

# [IMPORTANT]第一层聚类：内外侧点云识别分类
# X = np.concatenate((points, normals*50), axis=1)
# X = np.concatenate((points, normals*50, cos_sum.reshape(-1, 1)), axis=1)
# X = np.concatenate((points, cos_sum.reshape(-1, 1)), axis=1)
X = np.array(cos_sum).reshape(-1, 1)
# X就是数据点

# K-Means
# Define function:
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters)

# Fit the model:
km = kmeans.fit(X)
km_labels = km.labels_

# # Visualise results(3D):
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 假设三维数组中的数据是坐标点
# x = X[:, 0].flatten()
# y = X[:, 1].flatten()
# z = X[:, 2].flatten()
#
# ax.scatter(x, y, z, c=kmeans.labels_)
# ax.scatter(kmeans.cluster_centers_[:, 0],
#            kmeans.cluster_centers_[:, 1],
#            kmeans.cluster_centers_[:, 2],
#            marker='^', s=100, linewidth=2,
#            c=list(range(n_clusters)))
#
# plt.show()
#

# 用pyvista可视化点云
# 仅仅显示部分类别
witness_cluster = [0]  # 哪些类别要显示，如果是空列表就是全显示
if len(witness_cluster) != 0:
    witness_points = []  # 显示的点的所有编号
    for p in range(len(points)):
        if km_labels[p] in witness_cluster:
            witness_points.append(p)
    points = points[witness_points]
    km_labels = km_labels[witness_points]

# 创建点云
cloud = pv.PolyData(points)
# 为点云赋予类别颜色
clr = pca.generate_colors(km_labels)
cloud.point_data["colors"] = clr/255
print(np.array([km_labels, km_labels, km_labels]).shape)
# Plot the point cloud
plotter = pv.Plotter()
plotter.add_points(cloud, rgb=True, point_size=3)
# plotter.set_background('black')
plotter.add_axes()
plotter.show()
#
# # [IMPORTANT]第二层聚类：分面识别
# X2 = np.concatenate((points, normals[witness_points]*50), axis=1)   # X就是数据点
#
# # K-Means
# # Define function:
# n_clusters_l2 = 3
# kmeans = KMeans(n_clusters=n_clusters_l2)
#
# # Fit the model:
# km_l2 = kmeans.fit(X2)
# km_labels_l2 = km_l2.labels_
#
# # 用pyvista可视化点云
# # 创建点云
# cloud_l2 = pv.PolyData(points)
# # 为点云赋予类别颜色
# clr_l2 = pca.generate_colors(km_labels_l2)
# cloud_l2.point_data["colors"] = clr_l2/255
# # Plot the point cloud
# plotter = pv.Plotter()
# plotter.add_points(cloud_l2, rgb=True, point_size=3)
# # plotter.set_background('black')
# plotter.add_axes()
# plotter.show()