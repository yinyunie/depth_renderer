import argparse
from multiprocessing import Pool
import glob
import os
from functools import partial
import trimesh
import numpy as np
from external.utils import binvox_rw, voxels
from external.utils.libmesh import check_mesh_contains

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')
parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')
parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--pointcloud_size', type=int, default=100000,
                    help='Size of point cloud.')
parser.add_argument('--points_size', type=int, default=100000,
                    help='Size of points.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')
parser.add_argument('--packbits', action='store_true',
                help='Whether to save truth values as bit array.')
parser.add_argument('--bbox_in_folder', type=str,
                    help='Path to other input folder to extract'
                         'bounding boxes.')

def main(args):
    input_files = glob.glob(os.path.join(args.in_folder, '*', '*','model.off'))
    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), input_files)
    else:
        for p in input_files:
            process_path(p, args)

def process_path(in_path, args):
    mesh = trimesh.load(in_path, process=False)
    output_dir = os.path.dirname(in_path)

    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        if args.bbox_in_folder is not None:
            in_path_tmp = os.path.join(args.bbox_in_folder, *in_path.split('/')[3:5], 'model.obj')
            mesh_tmp = trimesh.load(in_path_tmp, process=False)
            bbox = mesh_tmp.bounding_box.bounds
        else:
            bbox = mesh.bounding_box.bounds

        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

        # Transform input mesh
        mesh.apply_translation(-loc)
        mesh.apply_scale(1 / scale)

    # Expert various modalities
    pointcloud_file = os.path.join(output_dir, 'pointcloud.npz')
    export_pointcloud(mesh, pointcloud_file, loc, scale, args)

    voxels_file = os.path.join(output_dir, 'voxels.binvox')
    export_voxels(mesh, voxels_file, loc, scale, args)

    points_file = os.path.join(output_dir, 'points.npz')
    export_points(mesh, points_file, loc, scale, args)

    mesh_file = os.path.join(output_dir, 'mesh.off')
    export_mesh(mesh, mesh_file, loc, scale, args)

def export_mesh(mesh, mesh_file, loc, scale, args):
    if not args.overwrite and os.path.exists(mesh_file):
        print('Mesh already exist: %s' % mesh_file)
        return

    print('Writing mesh: %s' % mesh_file)
    mesh.export(mesh_file)

def export_points(mesh, points_file, loc, scale, args):
    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot sample points.' % points_file)
        return

    if not args.overwrite and os.path.exists(points_file):
        print('Points already exist: %s' % points_file)
        return

    n_points_uniform = int(args.points_size * args.points_uniform_ratio)
    n_points_surface = args.points_size - n_points_uniform

    boxsize = 1 + args.points_padding
    points_uniform = np.random.rand(n_points_uniform, 3)
    points_uniform = boxsize * (points_uniform - 0.5)
    points_surface = mesh.sample(n_points_surface)
    points_surface += args.points_sigma * np.random.randn(n_points_surface, 3)
    points = np.concatenate([points_uniform, points_surface], axis=0)

    occupancies = check_mesh_contains(mesh, points)

    occupancies = np.packbits(occupancies)
    print('Writing points: %s' % points_file)
    np.savez_compressed(points_file, points=points, occupancies=occupancies,
                        loc=loc, scale=scale)

def export_voxels(mesh, voxels_file, loc, scale, args):

    if not mesh.is_watertight:
        print('Warning: mesh %s is not watertight!'
              'Cannot create voxelization.' % voxels_file)
        return

    if not args.overwrite and os.path.exists(voxels_file):
        print('Voxels already exist: %s' % voxels_file)
        return

    res = args.voxels_res
    voxels_occ = voxels.voxelize_interior(mesh, res)

    voxels_out = binvox_rw.Voxels(voxels_occ, (res,) * 3,
                                  translate=loc, scale=scale,
                                  axis_order='xyz')
    print('Writing voxels: %s' % voxels_file)
    with open(voxels_file, 'w') as f:
        voxels_out.write(f)


def export_pointcloud(mesh, output_file, loc, scale, args):
    if not args.overwrite and os.path.exists(output_file):
        print('Pointcloud already exist: %s' % output_file)
        return

    points, face_idx = mesh.sample(args.pointcloud_size, return_index = True)
    normals = mesh.face_normals[face_idx]

    print('Writing pointcloud: %s' % output_file)
    np.savez_compressed(output_file, points=points, normals=normals, loc=loc, scale=scale)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)