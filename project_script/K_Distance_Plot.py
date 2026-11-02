import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

# ── Global font / style settings ─────────────────────────────────────────────
rcParams['font.family']       = 'DejaVu Sans'
rcParams['axes.linewidth']    = 5
rcParams['xtick.major.width'] = 5
rcParams['ytick.major.width'] = 5
rcParams['xtick.major.size']  = 8
rcParams['ytick.major.size']  = 8

# Example usage:
#ply_path = r"H:\CreateBIM\dataset\d2k\d2k_1_subsampled_20cm.ply"
#ply_path = r"H:\CreateBIM\dataset\d2k\d2k_1_original.ply"
#ply_path = r"H:\CreateBIM\dataset\d2k\d2k_2_original.ply"
#ply_path = r"H:\CreateBIM\dataset\d2k\d2k_2_subsampled_10cm.ply"
#ply_path = r"H:\CreateBIM\dataset\d2k\d2k_5.ply"
#ply_path = r"F:\CreateBIM\dataset\d2k\d2k_1_original_transfer.ply"
#ply_path = r"F:\CreateBIM\dataset\d2k\complex_1_transfer.ply"
#ply_path = r"F:\CreateBIM\dataset\d2k\complex_20cm_voxel_transfer.ply"
#ply_path = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_transfer.ply"
ply_path = r"F:\CreateBIM\dataset\final_dataset\Experiment_2\experiment_2_transfer.ply"


pcd = o3d.io.read_point_cloud(ply_path)
xyz = np.asarray(pcd.points)


def k_distance_plot(xyz: np.ndarray, k: int = 50,
                    sample: int | None = None, seed: int = 0,
                    save_path: str | None = None):

    if sample is not None and xyz.shape[0] > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=sample, replace=False)
        xyz = xyz[idx]

    nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(xyz)
    distances, _ = nn.kneighbors(xyz)

    k_dist        = distances[:, -1]
    k_dist_sorted = np.sort(k_dist)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white', dpi=300)
    ax.set_facecolor('white')

    ax.plot(k_dist_sorted, color='#2176AE', linewidth=2.2, zorder=3)

    # ── Labels ────────────────────────────────────────────────────────────────
    ax.set_xlabel("Points sorted by distance",
                  fontsize=28, fontweight='normal',
                  color='#111111', labelpad=10)
    ax.set_ylabel(f"Distance to {k}-th\nnearest neighbor",
                  fontsize=28, fontweight='normal',
                  color='#111111', labelpad=10)
    # ax.set_title(f"K-distance plot for Dataset 1",  #todo
    #              fontsize=28, fontweight='normal',
    #              color='#111111', pad=12)

    # ── Ticks ─────────────────────────────────────────────────────────────────
    ax.tick_params(axis="both", which="major",
                   labelsize=24, width=1,
                   length=6, colors='#111111')

    # ── Grid ──────────────────────────────────────────────────────────────────
    ax.yaxis.grid(True, linestyle='--', alpha=0.40,
                  color='#AAAAAA', linewidth=0.9, zorder=0)
    ax.xaxis.grid(True, linestyle='--', alpha=0.40,
                  color='#AAAAAA', linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)

    # ── Spines ────────────────────────────────────────────────────────────────
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#666666')
        ax.spines[spine].set_linewidth(1.4)

    plt.tight_layout()

    # ── Save high-DPI PNG ─────────────────────────────────────────────────────
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', format='png')
        print(f"[INFO] Saved high-DPI figure → {save_path}")

    plt.show()

    return k_dist_sorted


k_dist_sorted = k_distance_plot(
    xyz,
    k         = 6,
    sample    = 2000,
    save_path = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\kdistance_dataset5.png"
)