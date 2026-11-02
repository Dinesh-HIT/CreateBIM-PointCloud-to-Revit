from pathlib import Path

#input_ply = Path(r"F:\CreateBIM\dataset\d2k\d2k_1_original.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\d2k\d2k_2_original.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\d2k\d2k_3.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\d2k\Complex_BD_Axci.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\d2k\complex_20cm_voxel.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\d2k\complex\complex_15cm_budi1.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\complex_final_20cm_voxel.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\D1.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\D1_20cm_voxel.ply")
#input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\data_final.ply")

#input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\data_final_1.ply")

#todo data final

# input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\data_final_2.ply")
# output_ply = Path(r"F:\CreateBIM\dataset\final_dataset\data_final_2_transfer.ply")


#TODO EXPERIMENT

# input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel.ply")
# output_ply = Path(r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_transfer.ply")

input_ply = Path(r"G:\CreateBIM\dataset\dataset4\experiment_1_window_points.ply")
output_ply = Path(r"G:\CreateBIM\dataset\dataset4\experiment_1_window_transfer.ply")


# input_ply = Path(r"G:\CreateBIM\dataset\dataset2\d2k_2_window_3_points.ply")
# output_ply = Path(r"G:\CreateBIM\dataset\dataset2\d2k_2_window_3_transfer.ply")

# input_ply = Path(r"G:\CreateBIM\dataset\dataset3\d2k_3_window_transfer.ply")
# output_ply = Path(r"G:\CreateBIM\dataset\dataset3\d2k_3_window_transfer_again.ply")

#todo: point support
#
# input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_voting.ply")
# output_ply = Path(r"F:\CreateBIM\dataset\final_dataset\Experiment\experiment_voting_transfer.ply")

# input_ply = Path(r"F:\CreateBIM\dataset\final_dataset\density_point.ply")
# output_ply = Path(r"F:\CreateBIM\dataset\final_dataset\density_point_transfer.ply")
#todo==end------=========================edge voting

#output_ply = Path(r"F:\CreateBIM\dataset\d2k\d2k_1_original_transfer.ply")
#output_ply = Path(r"F:\CreateBIM\dataset\d2k\d2k_2_transfer.ply")
#output_ply = Path(r"F:\CreateBIM\dataset\d2k\d2k_3_transfer.ply")
#output_ply = Path(r"F:\CreateBIM\dataset\final_dataset\D1_transfer.ply")



#output_ply = Path(r"F:\CreateBIM\dataset\final_dataset\data_final_2_transfer.ply")


# Read file
with open(input_ply, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

# Split header and point data
header = []
data_start = None
for i, line in enumerate(lines):
    header.append(line)
    if line.strip() == "end_header":
        data_start = i + 1
        break

if data_start is None:
    raise ValueError("Invalid PLY: end_header not found")

point_lines = lines[data_start:]

# Read XYZ from the first 3 columns
points = []
for line in point_lines:
    parts = line.strip().split()
    if len(parts) < 3:
        continue
    x, y, z = map(float, parts[:3])
    points.append((x, y, z))

if not points:
    raise ValueError("No point data found")

# Compute min corner
min_x = min(p[0] for p in points)
min_y = min(p[1] for p in points)
min_z = min(p[2] for p in points)

print("Shift applied:")
print("min_x =", min_x)
print("min_y =", min_y)
print("min_z =", min_z)

# Rewrite point cloud with shifted coordinates
new_point_lines = []
for line in point_lines:
    parts = line.strip().split()
    if len(parts) < 3:
        continue

    x = float(parts[0]) - min_x
    y = float(parts[1]) - min_y
    z = float(parts[2]) - min_z

    # keep all remaining attributes unchanged
    parts[0] = f"{x:.6f}"
    parts[1] = f"{y:.6f}"
    parts[2] = f"{z:.6f}"

    new_point_lines.append(" ".join(parts) + "\n")

# Update comment lines if you want
new_header = []
for line in header:
    s = line.strip()
    if s.startswith("comment minx"):
        new_header.append("comment minx 0.0\n")
    elif s.startswith("comment miny"):
        new_header.append("comment miny 0.0\n")
    elif s.startswith("comment minz"):
        new_header.append("comment minz 0.0\n")
    elif s.startswith("comment maxx"):
        new_header.append(f"comment maxx {max(p[0] for p in points) - min_x}\n")
    elif s.startswith("comment maxy"):
        new_header.append(f"comment maxy {max(p[1] for p in points) - min_y}\n")
    elif s.startswith("comment maxz"):
        new_header.append(f"comment maxz {max(p[2] for p in points) - min_z}\n")
    else:
        new_header.append(line)

# Save new file
with open(output_ply, "w", encoding="utf-8") as f:
    f.writelines(new_header)
    f.writelines(new_point_lines)

print(f"Saved shifted PLY to: {output_ply}")