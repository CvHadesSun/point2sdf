
import trimesh
import numpy as np
import torch
import cuda_kdtree
from .tools import normalize_mesh,load_mesh_and_sample,normalize_mesh_cpu,compute_angles_between_vectors
import time
from pykdtree.kdtree import KDTree
from skimage import measure

def compute_sdf(query_pts,kd_tree,src_pts,sign_field):
    '''input points and compute sdf.
    @input:
        pts: points for compute sdf.
        kd_tree: kd_tree for near surface points.
        sign_field: sparse sign field.
    @output:
        return: sdf for input points.
    '''

    pts = torch.clamp(query_pts,0.001,0.999)
    dist_tensor,_ = cuda_kdtree.query(pts,src_pts,kd_tree)
    res = sign_field.size(0)

    scaled_pts = pts  * res
    scaled_pts = torch.floor(scaled_pts).long()
    
    sign_result = sign_field[scaled_pts[:,0],scaled_pts[:,1],scaled_pts[:,2]]
    
    inside_mask = torch.where(sign_result < 0.0)
    dist_tensor[inside_mask] *= -1

    pts = pts * 2 - 1 # [-1,1]
    occ = torch.ones_like(sign_result).cuda()
    occ[inside_mask] = 0


    pts_occ_sdf = torch.cat([pts,occ.reshape(-1,1),dist_tensor.reshape(-1,1)],-1) # [xyz,occ,sdf]

    return pts_occ_sdf

def compute_sdf_from_normal(query_pts,kd_tree_box,data_ptr,src_pts,face_normals):
    '''input points and compute sdf.
    @input:
        pts: points for compute sdf.
        kd_tree: kd_tree for near surface points.
        sign_field: sparse sign field.
    @output:
        return: sdf for input points.
    '''


    dist_tensor,inds,ori_inds = cuda_kdtree.query_from_kdtree(query_pts, data_ptr,src_pts.size(0),kd_tree_box)
    indices = ori_inds[inds]

    neighbor_normals = face_normals[indices]
    neighbor_face_centers = src_pts[indices]
    vectors = query_pts - neighbor_face_centers

    dot_products = torch.sum(vectors * neighbor_normals,dim=-1)

    inside_mask = torch.where(dot_products <0.0)

    occ = torch.ones_like(dist_tensor).cuda()
    dist_tensor[inside_mask] *= -1
    occ[inside_mask] = 0
    pts = query_pts * 2 - 1 # [-1,1]

    pts_occ_sdf = torch.cat([pts,occ.reshape(-1,1),dist_tensor.reshape(-1,1)],-1) # [xyz,occ,sdf]

    return pts_occ_sdf


def compute_sdf_cpu(query_pts,kd_tree,sign_field):
    '''input points and compute sdf.
    @input:
        pts: points for compute sdf.
        kd_tree: kd_tree for near surface points.
        sign_field: sparse sign field.
    @output:
        return: sdf for input points.
    '''

    pts = torch.clamp(query_pts,0.001,0.999)
    np_pts = pts.cpu().numpy()
    dist, _ = kd_tree.query(np_pts, k=1)
    dist_tensor = torch.from_numpy(dist).float().cuda()
    res = sign_field.size(0)

    scaled_pts = pts  * res
    scaled_pts = torch.floor(scaled_pts).long()
    
    sign_result = sign_field[scaled_pts[:,0],scaled_pts[:,1],scaled_pts[:,2]]
    inside_mask = torch.where(sign_result < 0.5)
    dist_tensor[inside_mask] *=-1
    pts = pts * 2 - 1 # [-1,1]

    pts_occ_sdf = torch.cat([pts,sign_result.reshape(-1,1),dist_tensor.reshape(-1,1)],-1) # [xyz,occ,sdf]

    return pts_occ_sdf


def compute_sparse_sign_field(tris,res=512):
    '''input mesh and use cumesh2sdf to compute sparse sign field.
    @input:
        mesh_dir: path to mesh file.
        res: resolution of sdf.
    @output:
        return: sparse sign field.
    '''
    # band = 8/res
    # sparse_sdf = torchcumesh2sdf.get_sdf(tris, res, band) #+ 2/res # todo: can set batch size.
    # sparse_sign_field = torch.zeros_like(sparse_sdf)

    # outside_mask = torch.where(sparse_sdf > 0)
    # sparse_sign_field[outside_mask] = 1
    # near_idx_1 = torch.where(sparse_sdf < 10)
    # near_idx_0 = torch.where(sparse_sdf > -10)
    # near_sdf_0 = sparse_sdf[near_idx_0]
    # near_sdf_1 = sparse_sdf[near_idx_1]

    # import ipdb; ipdb.set_trace()

    # verts, faces, normals, values = measure.marching_cubes(sparse_sign_field.cpu().numpy(), level=0.5)
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # mesh.export('test.obj')
    return None

def compute_sdf_and_occ_cpu(mesh_dir,out_dir,count,res=512,epsilon=0.01):

    mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)
    normalize_mesh_cpu(mesh,band=8/res) 

    tris = np.array(mesh.triangles, dtype=np.float32, subok=False)
    tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')

    sparse_sign_field = compute_sparse_sign_field(tris,res=res)
    surface_points,_ = trimesh.sample.sample_surface(mesh, count)
    kd_tree= KDTree(surface_points)
    surface_points_tensor = torch.from_numpy(surface_points).float().cuda() + torch.randn(count, 3).float().cuda() * epsilon
    surface_pts_occ_sdf = compute_sdf_cpu(surface_points_tensor,kd_tree,sparse_sign_field) # [count,5]

    volume_points_tensor = (torch.rand(count,3)).float().cuda()
    volume_pts_occ_sdf = compute_sdf_cpu(volume_points_tensor,kd_tree,sparse_sign_field) # [count,5]
    final_all = torch.cat([surface_pts_occ_sdf,volume_pts_occ_sdf],0)

    np.save(out_dir,surface_pts_occ_sdf.cpu().numpy())


def compute_sdf_and_occ_cpu_input_mesh(mesh,out_dir,count,res=512,epsilon=0.01):

    # mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)
    # normalize_mesh(mesh,band=8/res) 

    sparse_sign_field = compute_sparse_sign_field(mesh,res=res)
    surface_points,_ = trimesh.sample.sample_surface(mesh, count)
    kd_tree= KDTree(surface_points)
    surface_points_tensor = torch.from_numpy(surface_points).float().cuda() + torch.randn(count, 3).float().cuda() * epsilon
    surface_pts_occ_sdf = compute_sdf_cpu(surface_points_tensor,kd_tree,sparse_sign_field) # [count,5]

    volume_points_tensor = (torch.rand(count,3)).float().cuda()
    volume_pts_occ_sdf = compute_sdf(volume_points_tensor,kd_tree,sparse_sign_field) # [count,5]
    final_all = torch.cat([surface_pts_occ_sdf,volume_pts_occ_sdf],0)

    # np.save(out_dir,final_all.cpu().numpy())


def compute_sdf_and_occ_points(mesh_dir,out_dir,count,res=512,epsilon=0.01,gpu=True):
    '''input mesh and compute sdf and occupancy for volume and surface points.
    @input:
        mesh_dir: path to mesh file.
        res: resolution of sdf.
        epsilon: epsilon for near surface points.
    @output:
        return: sdf and occupancy for volume and surface points.
    '''

    if not gpu:
        print('cpu')
        compute_sdf_and_occ_cpu(mesh_dir,out_dir,count,res,epsilon)
        return
    
    # mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)
    # normalize_mesh(mesh,band=8/res) 
    src_points_tensor,tris=load_mesh_and_sample(mesh_dir,100_000_000)
    src_points_tensor=src_points_tensor

    sparse_sign_field = compute_sparse_sign_field(tris,res=res)

    kd_tree = cuda_kdtree.build_kdtree(src_points_tensor)

    surface_points_sample,_ = load_mesh_and_sample(mesh_dir,count)

    surface_points_tensor = surface_points_sample + torch.randn(count, 3).float().cuda() * epsilon
    surface_pts_occ_sdf = compute_sdf(surface_points_tensor,kd_tree,src_points_tensor,sparse_sign_field) # [count,5]
    volume_points_tensor = (torch.rand(count,3)).float().cuda()
    # import ipdb; ipdb.set_trace()
    volume_pts_occ_sdf = compute_sdf(volume_points_tensor,kd_tree,src_points_tensor,sparse_sign_field) # [count,5]

    final_all = torch.cat([surface_pts_occ_sdf,volume_pts_occ_sdf],0)

    np.save(out_dir,final_all.cpu().numpy())


def compute_sdf_and_occ_points_input_mesh(mesh,out_dir,count,res=512,epsilon=0.01,gpu=True):
    '''input mesh and compute sdf and occupancy for volume and surface points.
    @input:
        mesh_dir: path to mesh file.
        res: resolution of sdf.
        epsilon: epsilon for near surface points.
    @output:
        return: sdf and occupancy for volume and surface points.
    '''

    if not gpu:
        compute_sdf_and_occ_cpu_input_mesh(mesh,out_dir,count,res,epsilon)
        return
    


    sparse_sign_field = compute_sparse_sign_field(mesh,res=res)
    surface_points,_ = trimesh.sample.sample_surface(mesh, count)
    # kd_tree= KDTree(surface_points)
    src_points_tensor = torch.from_numpy(surface_points).float().cuda()
    kd_tree = cuda_kdtree.build_kdtree(src_points_tensor)


    surface_points_tensor = torch.from_numpy(surface_points).float().cuda() + torch.randn(count, 3).float().cuda() * epsilon
    surface_pts_occ_sdf = compute_sdf(surface_points_tensor,kd_tree,src_points_tensor,sparse_sign_field) # [count,5]
    volume_points_tensor = (torch.rand(count,3)).float().cuda()
    # import ipdb; ipdb.set_trace()
    volume_pts_occ_sdf = compute_sdf(volume_points_tensor,kd_tree,src_points_tensor,sparse_sign_field) # [count,5]

    final_all = torch.cat([surface_pts_occ_sdf,volume_pts_occ_sdf],0)

    # np.save(out_dir,final_all.cpu().numpy())

def compute_sdf_and_occ_points_new(mesh_dir,out_dir,count,epsilon=0.01):
    '''input mesh and compute sdf and occupancy for volume and surface points.
    @input:
        mesh_dir: path to mesh file.
        res: resolution of sdf.
        epsilon: epsilon for near surface points.
    @output:
        return: sdf and occupancy for volume and surface points.
    '''


    src_points_tensor,_,face_normals=load_mesh_and_sample(mesh_dir,10_000_000)
    # import ipdb; ipdb.set_trace()
    points_ind = torch.arange(src_points_tensor.size(0), dtype=torch.int32, device='cuda:0')
    kd_tree_box,data_ptr = cuda_kdtree.build_kdtree_with_indices(src_points_tensor,points_ind)

    surface_points_sample,_,_= load_mesh_and_sample(mesh_dir,count)
    surface_points_tensor = surface_points_sample + torch.randn(count, 3).float().cuda() * epsilon
    surface_pts_occ_sdf = compute_sdf_from_normal(surface_points_tensor,kd_tree_box,data_ptr,src_points_tensor,face_normals) # [count,5]
    
    volume_points_tensor = (torch.rand(count,3)).float().cuda()
    volume_pts_occ_sdf = compute_sdf_from_normal(volume_points_tensor,kd_tree_box,data_ptr,src_points_tensor,face_normals) # [count,5]

    final_all = torch.cat([surface_pts_occ_sdf,volume_pts_occ_sdf],0)

    np.save(out_dir,final_all.cpu().numpy())


def compute_sdf_from_normal_cpu(query_pts,kd_tree,src_pts,face_normals):
    '''input points and compute sdf.
    @input:
        pts: points for compute sdf.
        kd_tree: kd_tree for near surface points.
        sign_field: sparse sign field.
    @output:
        return: sdf for input points.
    '''

    # pts = torch.clamp(query_pts,0.001,0.999)
    dist, indices = kd_tree.query(query_pts, k=1)

    neighbor_normals = face_normals[indices]
    neighbor_face_centers = src_pts[indices]
    vectors = query_pts - neighbor_face_centers

    dot_products = np.sum(vectors * neighbor_normals,axis=-1)

    inside_mask = np.where(dot_products <0.0)
    # import ipdb; ipdb.set_trace()

    occ = np.ones_like(dist)
    dist[inside_mask] *= -1
    occ[inside_mask] = 0
    pts = query_pts * 2 - 1 # [-1,1]

    pts_occ_sdf = np.concatenate([pts,occ.reshape(-1,1),dist.reshape(-1,1)],-1) # [xyz,occ,sdf]


    return pts_occ_sdf

def compute_sdf_and_occ_points_new_cpu(mesh_dir,out_dir,count,epsilon=0.01):
    mesh = trimesh.load(mesh_dir, process=False, force='mesh', skip_materials=True)
    normalize_mesh_cpu(mesh) 
    surface_pts,face_idx = trimesh.sample.sample_surface(mesh, 10_000_000)
    face_normals = mesh.face_normals[face_idx]

    kd_tree = KDTree(surface_pts)
    surface_sample,_ = trimesh.sample.sample_surface(mesh,count)
    surface_sample = surface_sample + np.random.randn(count, 3) * epsilon

    surface_pts_occ_sdf = compute_sdf_from_normal_cpu(surface_sample,kd_tree,surface_pts,face_normals) # [count,5]

    volume_sample = np.random.rand(count,3)
    volume_pts_occ_sdf = compute_sdf_from_normal_cpu(volume_sample,kd_tree,surface_pts,face_normals) # [count,5]

    final_all = np.concatenate([surface_pts_occ_sdf,volume_pts_occ_sdf],0)

    # np.save(out_dir,final_all)



    
