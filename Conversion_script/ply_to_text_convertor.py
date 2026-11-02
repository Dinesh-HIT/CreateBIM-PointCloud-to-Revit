# import open3d as o3d
# import numpy as np
#
# # === Step 1: Load the .ply file ===
# ply_file_path = r"G:\POINTCLOUD-PLY\point_cloud\iteration_100\point_cloud.ply"
# pcd = o3d.io.read_point_cloud(ply_file_path)
#
# # === Step 2: Convert to NumPy array ===
# points = np.asarray(pcd.points)
#
# # If color exists (Nx6 or Nx7), include it
# if pcd.has_colors():
#     colors = np.asarray(pcd.colors)
#     data = np.hstack((points, colors))
#     header = "{:<12s}{:<12s}{:<12s}{:<12s}{:<12s}{:<12s}".format("x", "y", "z", "r", "g", "b")
#     fmt = "%-12.6f%-12.6f%-12.6f%-12.6f%-12.6f%-12.6f"
# else:
#     data = points
#     header = "{:<12s}{:<12s}{:<12s}".format("x", "y", "z")
#     fmt = "%-12.6f%-12.6f%-12.6f"
#
# # === Step 3: Save to text file ===
# txt_file_path = "npg_ply_text.txt"
# np.savetxt(txt_file_path, data, fmt=fmt, header=header, comments='')
#
# print(f"Saved to {txt_file_path}")
import open3d as o3d
import numpy as np
from pathlib import Path

# Settings
ply_file_path = r"G:\POINTCLOUD-PLY\point_cloud\iteration_100\point_cloud.ply"
txt_file_path = r"G:\POINTCLOUD-PLY\point_cloud\iteration_100\npg_ply_text.txt"
as_uint8_rgb = False      # False => saves r,g,b as 0..1 floats; True => saves r,g,b as 0..255 ints
include_normals = True    # save normals if available

# === Load ===
pcd = o3d.io.read_point_cloud(ply_file_path)
if pcd.is_empty():
    raise SystemExit(f"No points found in {ply_file_path}")

points = np.asarray(pcd.points)          # (N,3), float
has_colors = pcd.has_colors()
colors = np.asarray(pcd.colors) if has_colors else None  # (N,3), float in [0,1]

has_normals = include_normals and pcd.has_normals()
normals = np.asarray(pcd.normals) if has_normals else None  # (N,3), float

# === Build data & header ===
cols = [points]
header = ["x", "y", "z"]

if has_colors:
    if as_uint8_rgb:
        # convert to 0..255 integers
        colors255 = (np.clip(colors, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        # We'll write this manually below to keep integer formatting clean
        pass
    else:
        cols.append(colors.astype(np.float64))
        header += ["r", "g", "b"]

if has_normals:
    cols.append(normals.astype(np.float64))
    header += ["nx", "ny", "nz"]

# === Save ===
Path(txt_file_path).parent.mkdir(parents=True, exist_ok=True)

if has_colors and as_uint8_rgb:
    # Mixed float (xyz) + int (rgb) => write line-by-line to preserve types nicely
    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write("".join(f"{h:<12s}" for h in (header if has_normals else header)) + "\n")
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = colors[i]
            r, g, b = int(round(r*255)), int(round(g*255)), int(round(b*255))
            line = f"{x:12.6f}{y:12.6f}{z:12.6f}{r:12d}{g:12d}{b:12d}"
            if has_normals:
                nx, ny, nz = normals[i]
                line += f"{nx:12.6f}{ny:12.6f}{nz:12.6f}"
            f.write(line + "\n")
else:
    # All-float path (xyz [+ rgb float] [+ normals float]) — fast and simple
    data = np.hstack(cols)
    fmts = ["%-12.6f"] * data.shape[1]   # per-column fmt
    np.savetxt(
        txt_file_path,
        data,
        fmt=fmts,
        header="".join(f"{h:<12s}" for h in header),
        comments=""
    )

print(f"Saved to {txt_file_path}")
print(f"Points: {len(points)}, colors: {has_colors}, normals: {has_normals}")
