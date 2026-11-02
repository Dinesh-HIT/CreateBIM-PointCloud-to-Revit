import numpy as np

# === Step 1: Load the .npy file ===
npy_file_path = r"H:\CreateBIM\building_output\d2k_3_subsampled_20cm_point_clusters.npy"  # ← replace with your file
data = np.load(npy_file_path)  # Expected shape: (N, 4)

# === Step 2: Save as a .txt file ===
txt_file_path = "numpy_filtered_building31.txt"
np.savetxt(txt_file_path, data, fmt="%.6f", delimiter="\t")

print(f"Saved to {txt_file_path}")
