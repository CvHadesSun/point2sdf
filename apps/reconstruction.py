
import os
import numpy as np
import trimesh
from skimage import measure
from scipy.interpolate import griddata
import argparse


def sdf2mesh(sdf_dir,out_dir,type='sdf'):
    sdf = np.load(sdf_dir)
    points = sdf[:,:3]
    if type == 'sdf':
        sdf_values = sdf[:,-1]
        level_th=0.0
    else:
        sdf_values = sdf[:,-2]
        level_th=0.5

    grid_x, grid_y, grid_z = np.mgrid[-1:1:384j, -1:1:384j, -1:1:384j]
    # 将 SDF 值插值到规则网格
    grid_sdf = griddata(points, sdf_values, (grid_x, grid_y, grid_z), method='linear', fill_value=1.0)
    # 使用 Marching Cubes 提取等值面 (SDF = 0)
    verts, faces, normals, values = measure.marching_cubes(grid_sdf, level=level_th)

    # 使用 trimesh 创建网格
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # 保存网格为 .obj 文件
    mesh.export(out_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--type', type=str, choices = ['occ', 'sdf'], default='sdf')
    
    args = parser.parse_args()

    sdf2mesh(args.input,args.output,args.type)