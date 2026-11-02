# pip install plyfile
from plyfile import PlyData
import numpy as np

ply_path = r"H:\CreateBIM\building_output\clustered_building3.ply"
pd = PlyData.read(ply_path)

# Inspect available vertex properties
v = pd["vertex"].data
names = v.dtype.names
print("Vertex fields:", names)

cols = []
headers = []

# xyz (required)
for k in ("x", "y", "z"):
    cols.append(v[k])
    headers.append(k)

# colors (red/green/blue or r/g/b)
for rgb in (("red","green","blue"), ("r","g","b")):
    if all(k in names for k in rgb):
        for k in rgb:
            cols.append(v[k])
            headers.append(k)
        break

# normals (optional)
if all(k in names for k in ("nx","ny","nz")):
    for k in ("nx","ny","nz"):
        cols.append(v[k])
        headers.append(k)

# semantic/instance label (try several common keys)
label_keys = ["label","class","class_id","semantic","semantic_id",
              "category_id","instance_id"]
lbl = next((k for k in label_keys if k in names), None)
if lbl:
    cols.append(v[lbl])
    headers.append(lbl)

# stack and save
data = np.vstack(cols).T

# build fmt per column (ints -> %d, floats -> %.6f)
fmts = []
for i, col in enumerate(cols):
    fmts.append("%d" if np.issubdtype(col.dtype, np.integer) else "%.6f")

out_path = r"H:\CreateBIM\Conversion\ngp_1003.txt"
np.savetxt(out_path, data, fmt=fmts, header=" ".join(headers), comments="")
print("Saved to:", out_path)
