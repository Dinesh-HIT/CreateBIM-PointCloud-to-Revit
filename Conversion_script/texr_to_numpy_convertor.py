import numpy as np

# === Step 1: Load .txt file (skip header) ===
txt_file_path = r"H:\CreateBIM\simulate_building_pc\Dclustered_points_with_labels.txt"

# If using tab delimiter and formatted header
data = np.loadtxt(txt_file_path, skiprows=1, delimiter=None)  # auto-detects spaces/tabs

# === Step 2: Save as .npy file ===
npy_file_path = "Dclustered_points_with_labels1.npy"
np.save(npy_file_path, data)

print(f"Data shape: {data.shape}")
print(f"Saved to: {npy_file_path}")
