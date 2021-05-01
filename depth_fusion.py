'''
Generate watertight mesh from depth scans.
author: ynie
date: Jan, 2020
'''
import sys
sys.path.append('.')
import os
from external import pyfusion
from data_config import shapenet_rendering_path, watertight_mesh_path, camera_setting_path, total_view_nums, shapenet_path
from tools.read_and_write import read_exr, read_txt, read_json
import numpy as np
import mcubes
from multiprocessing import Pool
from functools import partial
from tools.utils import dist_to_dep
from settings import cpu_cores
from tools.read_and_write import load_data_path


voxel_res = 256
truncation_factor = 10

def process_mesh(obj_path, view_ids, cam_Ks, cam_RTs):
    '''
    script for prepare watertigt mesh for training
    :param obj (str): object path
    :param view_ids (N-d list): which view ids would like to render (from 1 to total_view_nums).
    :param cam_Ks (N x 3 x 3): camera intrinsic parameters.
    :param cam_RTs (N x 3 x 3): camera extrinsic parameters.
    :return:
    '''
    cat, model = obj_path.split('/')[3:5]

    '''Decide save path'''
    output_file = os.path.join(watertight_mesh_path, cat, model, 'model.off')

    if os.path.exists(output_file):
        return None

    if not os.path.exists(os.path.join(watertight_mesh_path, cat, model)):
        os.makedirs(os.path.join(watertight_mesh_path, cat, model))

    '''Begin to process'''
    obj_dir = os.path.join(shapenet_rendering_path, cat, model)
    dist_map_dir = [os.path.join(obj_dir, 'depth_{0:03d}.exr'.format(view_id)) for view_id in view_ids]

    dist_maps = read_exr(dist_map_dir)
    depth_maps = np.float32(dist_to_dep(dist_maps, cam_Ks, erosion_size = 2))

    cam_Rs = np.float32(cam_RTs[:, :, :-1])
    cam_Ts = np.float32(cam_RTs[:, :, -1])

    views = pyfusion.PyViews(depth_maps, cam_Ks, cam_Rs, cam_Ts)

    voxel_size = 1. / voxel_res
    truncation = truncation_factor * voxel_size
    tsdf = pyfusion.tsdf_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, truncation, False)
    mask_grid = pyfusion.projmask_gpu(views, voxel_res, voxel_res, voxel_res, voxel_size, False)
    tsdf[mask_grid == 0.] = truncation

    # rotate to the correct system
    tsdf = np.transpose(tsdf[0], [2, 1, 0])

    # To ensure that the final mesh is indeed watertight
    tsdf = np.pad(tsdf, 1, 'constant', constant_values=1e6)
    vertices, triangles = mcubes.marching_cubes(-tsdf, 0)
    # Remove padding offset
    vertices -= 1
    # Normalize to [-0.5, 0.5]^3 cube
    vertices /= voxel_res
    vertices -= 0.5

    mcubes.export_off(vertices, triangles, output_file)

if __name__ == '__main__':
    '''generate watertight meshes by patch'''
    all_objects = load_data_path(shapenet_path)

    '''camera views'''
    view_ids = range(1, total_view_nums + 1)

    '''load camera parameters'''
    cam_K = np.loadtxt(os.path.join(camera_setting_path, 'cam_K/cam_K.txt'))
    cam_Ks = np.stack([cam_K] * total_view_nums, axis=0).astype(np.float32)
    cam_RT_dir = [os.path.join(camera_setting_path, 'cam_RT', 'cam_RT_{0:03d}.txt'.format(view_id)) for view_id in view_ids]
    cam_RTs = read_txt(cam_RT_dir)

    p = Pool(processes=cpu_cores)
    p.map(partial(process_mesh, view_ids=view_ids, cam_Ks=cam_Ks, cam_RTs=cam_RTs), all_objects)
    p.close()
    p.join()