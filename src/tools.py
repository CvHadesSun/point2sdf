import numpy as np

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

def normalize_mesh_cpu(mesh,band=8/512):
    print("Scaling Parameters: ", mesh.bounding_box.extents)
    mesh.vertices -= mesh.bounding_box.centroid
    mesh.vertices /= np.max(mesh.bounding_box.extents / 2)
    
    verts = mesh.vertices

    bbox_min = verts.min(0)
    bbox_max = verts.max(0)

    scale = 0.98 / (bbox_max - bbox_min).max()

    mesh.apply_scale(scale)

    mesh.vertices = mesh.vertices + 0.5


def normalize_mesh(mesh: Meshes):
    verts = mesh.verts_packed()  # Get vertices as a flat tensor
    center = verts.mean(0)       # Calculate centroid
    verts -= center              # Center the mesh
    
    # Calculate bounding box dimensions and scaling factor
    bbox_min = verts.min(0)[0]
    bbox_max = verts.max(0)[0]
    scale = 0.90 / (bbox_max - bbox_min).max()
    
    # Scale and shift vertices
    verts *= scale
    verts += 0.5                 # Shift vertices by 0.5 for normalization

    # Update the mesh with normalized vertices
    mesh = mesh.update_padded(torch.unsqueeze(verts, 0))

    # print("Scaling Parameters: ", bbox_max - bbox_min)

    # Extract the triangles in the shape [n, 3, 3]
    faces = mesh.faces_packed()         # Get face indices
    triangles = verts[faces]            # Get triangle vertices

    return mesh, triangles


def normalize_mesh_unit(mesh: Meshes):
    verts = mesh.verts_packed()  # Get vertices as a flat tensor
    center = verts.mean(0)       # Calculate centroid
    verts -= center              # Center the mesh at the origin

    # Calculate bounding box dimensions and scaling factor
    bbox_min = verts.min(0)[0]
    bbox_max = verts.max(0)[0]
    scale = 2.0 / (bbox_max - bbox_min).max()  # Scale to fit within [-1, 1]

    # Scale vertices to fit within [-1, 1]
    verts *= scale

    # Update the mesh with normalized vertices
    mesh = mesh.update_padded(torch.unsqueeze(verts, 0))

    # Extract the triangles in the shape [n, 3, 3]
    faces = mesh.faces_packed()          # Get face indices
    triangles = verts[faces]             # Get triangle vertices

    return mesh, triangles

def load_mesh_and_sample(mesh_dir, num_samples,mode='unit_01'): # mode unit_01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = load_objs_as_meshes([mesh_dir], device=device)
    if mode=='unit_11':
        normalized_mesh, triangles = normalize_mesh_unit(mesh)
    elif mode=='unit_01':
        normalized_mesh, triangles = normalize_mesh(mesh)
    else:
        raise ValueError("Invalid mode. Choose either 'unit_01' or 'unit_11'")
    surface_points,surface_normals = sample_points_from_meshes(normalized_mesh, num_samples,return_normals=True)
    return surface_points[0],triangles[0],surface_normals[0]



def compute_angles_between_vectors(vectors_a, vectors_b):
    # Compute dot products between corresponding vectors
    dot_products = torch.sum(vectors_a * vectors_b, dim=1)
    
    # Compute the norms (magnitudes) of each vector
    norms_a = torch.norm(vectors_a, dim=1)
    norms_b = torch.norm(vectors_b, dim=1)
    
    # Compute the cosine of the angles
    cos_angles = dot_products / (norms_a * norms_b + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Clamp values to the valid range for arccos to avoid numerical errors
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    
    # Compute the angles in radians
    # angles = torch.acos(cos_angles)

    return cos_angles