import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import open3d as o3d
import building_model as bm

# Input:
#     data: point cloud, an Nx3 matrix
#     correlation: distinguishes np's cov and corrcoef; defaults to False if not provided
#     sort: sort the eigenvalues; sorting makes it convenient for other functions; defaults to True if not provided
# Output:
#     eigenvalues: eigenvalues
#     eigenvectors: eigenvectors


# This function computes the cosine distance
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

    # Load the point cloud
    points = pd.DataFrame(points)

    points.columns = ["x", "y", "z"]
    point_cloud_pynt = PyntCloud(points=points)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d])  # display the original point cloud

    # Adjust the point cloud color
    point_cloud_o3d.paint_uniform_color([0, 0.4, 0.6])

    # Get the points from the point cloud; only the points are processed
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # Use PCA to analyze the principal direction of the point cloud
    w, v = PCA(points)
    point_cloud_vector = v[:, 0]  # the vector corresponding to the principal direction of the point cloud
    print('the main orientation of this pointcloud is: ', point_cloud_vector)
    # Draw the three principal directions
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

    # Loop to compute the normal vector of each point
    if show_normals:
        pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
        normals = []
        for point in point_cloud_o3d.points:
            [k, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=1000)
            print(idx)
            w, v = PCA(points.iloc[idx, :])
            normals.append(v[:, 2])

        normals = np.array(normals, dtype=np.float64)*0.1
        # TODO: the normal vectors are stored in normals here
        point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
        # o3d.visualization.draw_geometries([point_cloud_o3d], point_show_normals=True)
        o3d.visualization.draw_geometries([point_cloud_o3d, line_set, axis_pcd], window_name='My Window', width=800,
                                          height=600, left=50, top=50, point_show_normal=True)


def get_normals(point_cloud, normalization=True, absolute_value=True):
    # Load the point cloud
    points = pd.DataFrame(point_cloud)
    points.columns = ["x", "y", "z"]
    point_cloud_pynt = PyntCloud(points=points)
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)

    # Loop to compute the normal vectors
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)

    normals = []  # this list stores the normal vector of each point
    nearest_list = []  # this list stores the indices of the n nearest neighbors of each point

    for point in point_cloud_o3d.points:
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point, knn=1000)
        w, v = PCA(points.iloc[idx, :])
        normals.append(v[:, 2])
        nearest_list.append(idx)

    normals = np.array(normals, dtype=np.float64)
    nearest_list = np.array(nearest_list, dtype=int)

    # Normalize and take the absolute value
    if normalization:
        # Compute the magnitude of the vectors
        magnitudes = np.sqrt(np.sum(normals ** 2, axis=1))
        # Normalize the vectors
        normals = normals / magnitudes[:, np.newaxis]

    if absolute_value:
        # Take the absolute value
        normals = np.abs(normals)

    # Compute the cosine of the angle between each point's normal and its neighbors' normals
    cos_matrix = []  # this matrix stores the angle (in radians) between each point's normal and those of its n nearest neighbors
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