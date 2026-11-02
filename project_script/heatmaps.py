import os
import csv
import numpy as np
import open3d as o3d
import pyvista as pv
import ifcopenshell
import ifcopenshell.geom
from PIL import Image, ImageDraw, ImageFont

# =========================================================
# USER INPUTS
# =========================================================
# POINT_CLOUD_PATH = r"F:\CreateBIM\dataset\d2k\d2k_1_original_transfer.ply"
# BIM_IFC_PATH     = r"F:\CreateBIM\dataset\heatmaps_IFC_model_input\data_1.ifc"
#
# POINT_CLOUD_PATH = r"F:\CreateBIM\dataset\d2k\d2k_2_transfer.ply"
# BIM_IFC_PATH     = r"F:\CreateBIM\dataset\heatmaps_IFC_model_input\data_2.ifc"

# POINT_CLOUD_PATH = r"F:\CreateBIM\dataset\d2k\d2k_3_transfer.ply"
# BIM_IFC_PATH     = r"F:\CreateBIM\dataset\heatmaps_IFC_model_input\automatic_dataset3.ifc"

POINT_CLOUD_PATH = r"F:\CreateBIM\dataset\final_dataset\Experiment_1\experiment_1_voxel_transfer.ply"
BIM_IFC_PATH     = r"F:\CreateBIM\dataset\Reference BIM transfer\Plus_sign_ref.ifc"

OUTPUT_DIR       = r"F:\CreateBIM\dataset\validation\heatmaps_plus_sign"
OUTPUT_IMAGE     = os.path.join(OUTPUT_DIR, "heatmap_pyvista.png")
OUTPUT_CSV       = os.path.join(OUTPUT_DIR, "heatmap_statistics.csv")
OUTPUT_TXT       = os.path.join(OUTPUT_DIR, "heatmap_statistics.txt")

MAX_DISPLAY_DIST = 0.15   # meters
POINT_SIZE       = 3.5
MESH_OPACITY     = 0.22   # 0=transparent, 1=opaque
SHOW_MESH        = True
WINDOW_SIZE      = (2600, 1600)

# 2D overlay scale bar
DRAW_SCALE_BAR   = True
SCALE_LABEL      = "10"   # shown on image only

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOADERS
# =========================================================
def load_point_cloud(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=float)
    if pts.size == 0:
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return pts


def load_ifc_mesh_arrays(path: str):
    ifc = ifcopenshell.open(path)

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    candidate_types = [
        "IfcWall", "IfcWallStandardCase",
        "IfcSlab", "IfcRoof", "IfcDoor", "IfcWindow",
        "IfcBeam", "IfcColumn", "IfcStair", "IfcRailing",
        "IfcBuildingElementProxy"
    ]

    products = []
    for t in candidate_types:
        try:
            products.extend(ifc.by_type(t))
        except Exception:
            pass

    if len(products) == 0:
        products = ifc.by_type("IfcProduct")

    for product in products:
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            geom = shape.geometry

            verts = np.array(geom.verts, dtype=np.float64).reshape(-1, 3)
            faces = np.array(geom.faces, dtype=np.int32).reshape(-1, 3)

            if verts.size == 0 or faces.size == 0:
                continue

            all_vertices.append(verts)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(verts)

        except Exception:
            continue

    if len(all_vertices) == 0 or len(all_faces) == 0:
        raise ValueError(f"No triangulated IFC geometry could be extracted from: {path}")

    vertices = np.vstack(all_vertices)
    faces = np.vstack(all_faces)
    return vertices, faces


def build_open3d_mesh(vertices: np.ndarray, faces: np.ndarray):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def build_pyvista_mesh(vertices: np.ndarray, faces: np.ndarray):
    face_cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).ravel()
    mesh = pv.PolyData(vertices, face_cells)
    return mesh


# =========================================================
# DISTANCE COMPUTATION
# =========================================================
def compute_point_to_mesh_distance(points_xyz: np.ndarray, mesh_o3d: o3d.geometry.TriangleMesh) -> np.ndarray:
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)
    query_points = o3d.core.Tensor(points_xyz.astype(np.float32))
    distances = scene.compute_distance(query_points).numpy()
    return distances


# =========================================================
# STATISTICS
# =========================================================
def compute_statistics(distances: np.ndarray) -> dict:
    distances = np.asarray(distances, dtype=float)
    stats = {
        "count": int(distances.size),
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(np.sqrt(np.mean(distances ** 2))),
        "p95_m": float(np.percentile(distances, 95)),
        "max_m": float(np.max(distances)),
        "min_m": float(np.min(distances)),
    }
    return stats


def save_statistics_csv(stats: dict, csv_path: str):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in stats.items():
            writer.writerow([k, v])


def save_statistics_txt(stats: dict, txt_path: str):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Heatmap validation statistics\n")
        f.write("============================\n")
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def print_statistics(stats: dict):
    print("Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# =========================================================
# CAMERA
# =========================================================
def set_camera_style(plotter: pv.Plotter, all_points: np.ndarray):
    center = all_points.mean(axis=0)
    bounds_min = all_points.min(axis=0)
    bounds_max = all_points.max(axis=0)
    diag = np.linalg.norm(bounds_max - bounds_min)

    eye = center + np.array([0.9, -0.7, 0.55]) * diag
    up = (0.0, 0.0, 1.0)

    plotter.camera_position = [tuple(eye), tuple(center), up]
    plotter.camera.zoom(0.60)


# =========================================================
# XYZ TRIAD
# =========================================================
def add_axes_triad(plotter: pv.Plotter):
    plotter.add_axes(
        xlabel="X",
        ylabel="Y",
        zlabel="Z",
        line_width=2,
        labels_off=False
    )


# =========================================================
# RENDER
# =========================================================
def render_with_pyvista(points_xyz: np.ndarray, distances: np.ndarray, mesh_pv: pv.PolyData | None, output_path: str):
    clipped = np.clip(distances, 0.0, MAX_DISPLAY_DIST)

    point_cloud = pv.PolyData(points_xyz)
    point_cloud["Distance to BIM mesh (m)"] = clipped

    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("white")

    if SHOW_MESH and mesh_pv is not None:
        plotter.add_mesh(
            mesh_pv,
            color=(0.82, 0.82, 0.82),
            opacity=MESH_OPACITY,
            smooth_shading=True,
            show_edges=False,
            lighting=True
        )

    plotter.add_mesh(
        point_cloud,
        scalars="Distance to BIM mesh (m)",
        cmap="jet",
        clim=[0.0, MAX_DISPLAY_DIST],
        point_size=POINT_SIZE,
        render_points_as_spheres=True,
        opacity=1.0,
        scalar_bar_args={
            "title": "Distance to BIM mesh (m)",
            "vertical": True,
            "width": 0.05,
            "height": 0.55,
            "position_x": 0.86,
            "position_y": 0.12,
            "title_font_size": 34,
            "label_font_size": 28,
            "font_family": "arial",
            "fmt": "%.2f",
            "n_labels": 6,
            "color": "black",
            "bold": False,
            "italic": False,
            "shadow": False
        }
    )

    add_axes_triad(plotter)
    set_camera_style(plotter, points_xyz)

    plotter.show(screenshot=output_path)
    plotter.close()


# =========================================================
# 2D SCALE BAR OVERLAY
# =========================================================
def add_2d_scale_overlay(image_path: str, scale_label: str = "10"):
    """
    Add a fixed horizontal 2D scale bar to the saved screenshot.
    This is screen-based, so it stays horizontal and clearly visible.
    """
    if not DRAW_SCALE_BAR:
        return

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    # position near lower-left corner, close to XYZ triad
    bar_x0 = int(w * 0.12)
    bar_y = int(h * 0.925)
    bar_len = int(w * 0.13)
    tick_h = 12

    # main bar
    draw.line((bar_x0, bar_y, bar_x0 + bar_len, bar_y), fill="black", width=3)

    # end ticks
    draw.line((bar_x0, bar_y - tick_h, bar_x0, bar_y + tick_h), fill="black", width=2)
    draw.line((bar_x0 + bar_len, bar_y - tick_h, bar_x0 + bar_len, bar_y + tick_h), fill="black", width=2)

    # centered label above bar
    bbox = draw.textbbox((0, 0), scale_label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = bar_x0 + (bar_len - text_w) // 2
    text_y = bar_y - text_h - 14
    draw.text((text_x, text_y), scale_label, fill="black", font=font)

    img.save(image_path)


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading point cloud...")
    points_xyz = load_point_cloud(POINT_CLOUD_PATH)

    print("Loading IFC mesh...")
    vertices, faces = load_ifc_mesh_arrays(BIM_IFC_PATH)

    print("Building meshes...")
    mesh_o3d = build_open3d_mesh(vertices, faces)
    mesh_pv = build_pyvista_mesh(vertices, faces)

    print("Computing distances...")
    distances = compute_point_to_mesh_distance(points_xyz, mesh_o3d)

    print("Computing statistics...")
    stats = compute_statistics(distances)
    print_statistics(stats)
    save_statistics_csv(stats, OUTPUT_CSV)
    save_statistics_txt(stats, OUTPUT_TXT)

    print("Rendering with PyVista...")
    render_with_pyvista(points_xyz, distances, mesh_pv if SHOW_MESH else None, OUTPUT_IMAGE)

    print("Adding 2D scale overlay...")
    add_2d_scale_overlay(OUTPUT_IMAGE, scale_label=SCALE_LABEL)

    print(f"Done. Figure saved to:\n{OUTPUT_IMAGE}")
    print(f"CSV saved to:\n{OUTPUT_CSV}")
    print(f"TXT saved to:\n{OUTPUT_TXT}")


if __name__ == "__main__":
    main()