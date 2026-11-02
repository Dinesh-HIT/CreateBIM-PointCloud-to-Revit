import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import open3d as o3d
import building_model as bm

# 输入：
#     data：点云，Nx3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量


# 这个函数用来计算余弦距离
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    cos_angle = np.clip(dot_product, -1, 1)
    angle_radians = np.arccos(cos_angle)
    return angle_radians


def PCA(data, sort=True):
    X = np.asarray(data).T
    X_mean = np.mean(X, axis=1).reshape(3, 1)
    X_head = X - X_mean
    H = X_head.dot(X_head.T)
    eigenvalues, eigenvectors = np.linalg.eig(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


def point_cloud_with_mo(points, show_normals=True):

    # 点云载入
    points = pd.DataFrame(points)

    points.columns = ["x", "y", "z"]
    point_cloud_pynt = PyntCloud(points=points)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

    # 调整点云颜色
    point_cloud_o3d.paint_uniform_color([0, 0.4, 0.6])

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points)
    point_cloud_vector = v[:, 0]  # 点云主方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # 绘制三个主方向
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 0] + v[:, 2]*100, [0, 0, 0] + v[:, 1]*100, [0, 0, 0] + v[:, 0]*100])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.add_geometry(point_cloud_o3d)
    vis.add_geometry(line_set)
    vis.add_geometry(axis_pcd)
    render_option = vis.get_render_option()
    render_option.point_size = 2.5
    vis.run()
    vis.destroy_window()

    # 循环计算每个点的法向量
    if show_normals:
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        normals = []
        for point in point_cloud_o3d.points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=1000)
            print(idx)
            w, v = PCA(points.iloc[idx, :])
            normals.append(v[:, 2])

        normals = np.array(normals, dtype=np.float64)*0.1
        # TODO: 此处把法向量存放在了normals中
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        # o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normals=True)
        o3d.visualization.draw_geometries([point_cloud_o3d, line_set, axis_pcd], window_name='My Window', width=800,
                                          height=600, left=50, top=50, point_show_normal=True)


def get_normals(point_cloud, normalization=True, absolute_value=True):
    # 点云载入
    points = pd.DataFrame(point_cloud)
    points.columns = ["x", "y", "z"]
    point_cloud_pynt = PyntCloud(points=points)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

    # 循环计算法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)

    normals = []  # 这个list存储了每个点的法向量
    nearest_list = []  # 这个list存储了每个点的最邻近n点的索引号

    for point in point_cloud_o3d.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=1000)
        w, v = PCA(points.iloc[idx, :])
        normals.append(v[:, 2])
        nearest_list.append(idx)

    normals = np.array(normals, dtype=np.float64)
    nearest_list = np.array(nearest_list, dtype=int)

    # 归一化和取绝对值
    if normalization:
        # 计算向量的模
        magnitudes = np.sqrt(np.sum(normals ** 2, axis=1))
        # 归一化向量
        normals = normals / magnitudes[:, np.newaxis]

    if absolute_value:
        # 取绝对值
        normals = np.abs(normals)

    # 计算每个点的法向量和其临近点法向量的夹角余弦值
    cos_matrix = []  # 这个矩阵存储了每个点和其最邻近n点之间法向量的夹角余弦弧度
    for i in range(len(normals)):
        normal = normals[i]
        near_point_normals = normals[nearest_list[i]]
        cos_list = []
        for j in range(len(near_point_normals)):
            cos = angle_between_vectors(normal, near_point_normals[j])
            cos_list.append(cos)
        cos_matrix.append(cos_list)
    cos_matrix = np.array(cos_matrix)
    cos_sum = np.sum(cos_matrix, axis=1)
    cos_var = np.var(cos_matrix, axis=1)

    return normals, cos_sum, cos_var


def generate_colors(km_labels):
    n_clusters = len(set(km_labels))
    clrs = np.empty((len(km_labels), 3))
    colors = np.random.randint(0, 256, size=(n_clusters+1, 3))

    for i in range(len(km_labels)):
        label = km_labels[i]
        clrs[i] = colors[label]

    return clrs


