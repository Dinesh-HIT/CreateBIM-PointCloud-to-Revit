import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt


def load_mesh(path):
    """
    Load an OBJ mesh safely across different trimesh versions.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mesh file not found:\n{path}")

    mesh = trimesh.load_mesh(path, process=True)

    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in: {path}")
        mesh = trimesh.util.concatenate(meshes)

    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {path}")

    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    try:
        mesh.merge_vertices()
    except Exception:
        pass

    try:
        mesh.fix_normals()
    except Exception:
        pass

    return mesh


def voxel_iou(mesh_a, mesh_b, pitch=20.0):
    """
    Compute voxel-based volumetric IoU.
    """
    vox_a = mesh_a.voxelized(pitch).fill()
    vox_b = mesh_b.voxelized(pitch).fill()

    idx_a = set(map(tuple, np.asarray(np.round(vox_a.points / pitch), dtype=np.int64)))
    idx_b = set(map(tuple, np.asarray(np.round(vox_b.points / pitch), dtype=np.int64)))

    inter = len(idx_a & idx_b)
    union = len(idx_a | idx_b)

    if union == 0:
        raise ValueError("Union voxel count is zero.")

    iou = inter / union
    return iou, inter, union, len(idx_a), len(idx_b)


def run_dataset(dataset_name, auto_path, ref_path, voxel_sizes, output_dir):
    """
    Run voxel IoU sensitivity for one dataset.
    """
    print("=" * 80)
    print(f"{dataset_name}")
    print("=" * 80)
    print("Auto exists:", os.path.exists(auto_path), auto_path)
    print("Ref exists :", os.path.exists(ref_path), ref_path)

    auto_mesh = load_mesh(auto_path)
    ref_mesh = load_mesh(ref_path)

    print("Mesh Quality Check")
    print(f"Automatic BIM watertight : {auto_mesh.is_watertight}")
    print(f"Reference BIM watertight : {ref_mesh.is_watertight}")
    print(f"Automatic BIM vertices   : {len(auto_mesh.vertices)}")
    print(f"Reference BIM vertices   : {len(ref_mesh.vertices)}")
    print(f"Automatic BIM faces      : {len(auto_mesh.faces)}")
    print(f"Reference BIM faces      : {len(ref_mesh.faces)}")
    print()

    rows = []
    iou_values = []

    for voxel_size in voxel_sizes:
        try:
            iou_voxel, inter, union, n_auto, n_ref = voxel_iou(
                auto_mesh, ref_mesh, pitch=float(voxel_size)
            )

            rows.append([voxel_size, iou_voxel, inter, union, n_auto, n_ref])
            iou_values.append(iou_voxel)

            print(f"Voxel size             : {voxel_size}")
            print(f"Automatic filled voxels: {n_auto}")
            print(f"Reference filled voxels: {n_ref}")
            print(f"Intersection voxels    : {inter}")
            print(f"Union voxels           : {union}")
            print(f"Voxel 3D IoU           : {iou_voxel:.6f}")
            print("-" * 80)

        except Exception as e:
            print(f"{dataset_name} failed at voxel size {voxel_size}: {e}")
            rows.append([voxel_size, np.nan, np.nan, np.nan, np.nan, np.nan])
            iou_values.append(np.nan)

    rows = np.array(rows, dtype=float)

    dataset_csv = os.path.join(output_dir, f"{dataset_name.lower().replace(' ', '_')}_voxel_iou.csv")
    np.savetxt(
        dataset_csv,
        rows,
        delimiter=",",
        header="Voxel_Size,Voxel_3D_IoU,Intersection_Voxels,Union_Voxels,Automatic_Filled_Voxels,Reference_Filled_Voxels",
        comments="",
        fmt="%.6f"
    )
    print(f"Saved dataset CSV: {dataset_csv}")
    print()

    return iou_values, rows


def main():
    output_dir = r"F:\CreateBIM\dataset\IOU"
    os.makedirs(output_dir, exist_ok=True)

    voxel_sizes = list(range(50, 101, 10))   # 20, 30, ..., 100

    datasets = [
        (
            "Dataset 1",
            r"F:\CreateBIM\dataset\IOU\model_BIM_1.obj",
            r"F:\CreateBIM\dataset\IOU\Ref_BIM_1_transfer.obj"
        ),
        (
            "Dataset 2",
            r"F:\CreateBIM\dataset\IOU\model_BIM_2.obj",
            r"F:\CreateBIM\dataset\IOU\Ref_BIM_2_transfer.obj"
        ),
        (
            "Dataset 3",
            r"F:\CreateBIM\dataset\IOU\model_BIM_3.obj",
            r"F:\CreateBIM\dataset\IOU\Ref_BIM_3_transfer.obj"
        ),
    ]

    all_results = {}
    combined_rows = []

    for dataset_name, auto_path, ref_path in datasets:
        iou_values, rows = run_dataset(
            dataset_name, auto_path, ref_path, voxel_sizes, output_dir
        )
        all_results[dataset_name] = iou_values

        for r in rows:
            combined_rows.append([dataset_name, int(r[0]), r[1], r[2], r[3], r[4], r[5]])

    # Save combined CSV
    combined_csv = os.path.join(output_dir, "combined_voxel_iou_sensitivity.csv")
    with open(combined_csv, "w", encoding="utf-8") as f:
        f.write("Dataset,Voxel_Size,Voxel_3D_IoU,Intersection_Voxels,Union_Voxels,Automatic_Filled_Voxels,Reference_Filled_Voxels\n")
        for row in combined_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f}\n"
            )
    print(f"Saved combined CSV: {combined_csv}")

    # Plot all datasets on one figure
    plt.figure(figsize=(8, 5))

    for dataset_name, iou_values in all_results.items():
        plt.plot(voxel_sizes, iou_values, marker='o', label=dataset_name)

    plt.xlabel("Voxel Size (mm)")
    plt.ylabel("Voxel-Based 3D IoU")
    plt.title("Sensitivity of Voxel-Based 3D IoU to Voxel Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    combined_fig = os.path.join(output_dir, "combined_voxel_iou_sensitivity.png")
    plt.savefig(combined_fig, dpi=300, bbox_inches="tight")
    print(f"Saved combined figure: {combined_fig}")

    plt.show()


if __name__ == "__main__":
    main()