import trimesh
import numpy as np
from PIL import Image
import open3d as o3d

def sample_mesh_with_texture(obj_path, num_points=100000):
    # Cargar la malla con Trimesh (procesa .mtl automáticamente si está presente)
    scene_or_mesh = trimesh.load(obj_path, process=False)
    
    # Si se carga como una escena, extraer la malla principal
    if isinstance(scene_or_mesh, trimesh.Scene):
        # Combinar todas las geometrías en una sola malla
        mesh = trimesh.util.concatenate(
            [geom for geom in scene_or_mesh.geometry.values()]
        )
    else:
        mesh = scene_or_mesh



    # Verificar si la malla tiene coordenadas UV
    if mesh.visual.uv is None or len(mesh.visual.uv) == 0:
        raise ValueError("La malla no tiene coordenadas UV ni textura asignada.")

    # Obtener la textura como una imagen
    texture_image = mesh.visual.material.image
    if texture_image is None:
        raise ValueError("No se pudo cargar la textura desde el archivo .mtl.")

    # Convertir la textura a un arreglo numpy
    texture = np.array(texture_image) / 255.0  # Normalizar colores a rango [0, 1]

    # Muestrear puntos de la superficie de la malla
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # Obtener coordenadas UV correspondientes a los puntos muestreados
    uv_coords = mesh.visual.uv[mesh.faces[face_indices]]
    barycentric_coords = np.random.rand(len(points), 3)
    barycentric_coords /= barycentric_coords.sum(axis=1, keepdims=True)
    uv_sampled = np.einsum('ijk,ij->ik', uv_coords, barycentric_coords)

    # Convertir coordenadas UV a colores de la textura
    tex_coords = (uv_sampled * np.array([texture.shape[1], texture.shape[0]])).astype(int)
    # Limitar índices a los límites válidos de la textura
    tex_coords[:, 0] = np.clip(tex_coords[:, 0], 0, texture.shape[1] - 1)
    tex_coords[:, 1] = np.clip(tex_coords[:, 1], 0, texture.shape[0] - 1)
    colors = texture[tex_coords[:, 1], tex_coords[:, 0]]

    return points, colors

def save_point_cloud_with_colors(points, colors, out_path):
    if colors.shape[1] == 4:  # Si incluye componente alfa
        colors = colors[:, :3]  # Mantener solo R, G, B
    # Asegurar que las dimensiones sean compatibles
    if colors.shape[0] != points.shape[0]:
        raise ValueError("El número de puntos y colores no coincide.")
    if colors.shape[1] != 3:
        print(colors.shape[1])
        raise ValueError("Los colores deben tener una dimensión de (N, 3).")
    
    # Crear la nube de puntos con colores
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Guardar la nube de puntos
    o3d.io.write_point_cloud(out_path, pcd)
    print(f"Nube de puntos guardada en: {out_path}")

