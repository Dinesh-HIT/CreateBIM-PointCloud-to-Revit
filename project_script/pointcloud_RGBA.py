import numpy as np
import pyvista as pv


# 生成5000个随机点的坐标
num_points = 200000
points = np.random.rand(num_points, 3)

# 随机RGBA格式颜色（0到1之间）
colors = np.random.rand(num_points, 4)

# 修改RGB格式颜色（0到1之间）
colors[:, :3] = 0.8
# 让颜色沿着z轴变化
colors[:, 1] = points[:, 2]
# 让不透明度沿着x轴变化
colors[:, 3] = points[:, 0]


# Create a PyVista PolyData object
cloud = pv.PolyData(points)

# Assign random brightness to each point
cloud.point_data["colors"] = colors

# Plot the point cloud
plotter = pv.Plotter()
plotter.add_points(cloud, rgb=True, point_size=3)
plotter.set_background('black')
plotter.add_axes()
plotter.show()


