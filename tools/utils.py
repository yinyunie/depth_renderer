'''
General functions

author: ynie
date: Jan, 2020
'''

import os
import numpy as np
import point_cloud_utils as pcu
from tools.read_and_write import read_obj
from scipy.ndimage.morphology import grey_erosion

def append_dir(parent_dir, foldername, mode = 'i'):
    output = os.path.join(parent_dir, foldername)

    if mode == 'i':
        if not os.path.exists(output):
            raise FileExistsError('%s does not exist.' % (output))
    elif mode == 'o':
        if not os.path.exists(output):
            os.makedirs(output)
    else:
        raise ValueError("mode should be 'i' / 'o'.")

    return output

def normalize_points(in_points, padding = 0.1):
    '''
    normalize points into [-0.5, 0.5]^3, and output point set and scale info.
    :param in_points: input point set, Nx3.
    :return:
    '''
    vertices = in_points.copy()

    bb_min = vertices.min(0)
    bb_max = vertices.max(0)
    total_size = (bb_max - bb_min).max() / (1 - padding)

    # centroid
    centroid = (bb_min + bb_max)/2.
    vertices = vertices - centroid

    vertices = vertices/total_size

    return vertices, total_size, centroid

def normalize_obj_file(input_obj_file, output_obj_file, padding = 0.1):
    """
    normalize vertices into [-0.5, 0.5]^3, and write the result into another .obj file.
    :param input_obj_file: input .obj file
    :param output_obj_file: output .obj file
    :return:
    """
    vertices = read_obj(input_obj_file)['v']

    vertices, total_size, centroid = normalize_points(vertices, padding=padding)

    input_fid = open(input_obj_file, 'r', encoding='utf-8')
    output_fid = open(output_obj_file, 'w', encoding='utf-8')

    v_id = 0
    for line in input_fid:
        if line.strip().split(' ')[0] != 'v':
            output_fid.write(line)
        else:
            output_fid.write(('v' + ' %f' * len(vertices[v_id]) + '\n') % tuple(vertices[v_id]))
            v_id = v_id + 1

    output_fid.close()
    input_fid.close()

    return total_size, centroid

def dist_to_dep(dist_maps, cam_Ks, **kwargs):
    '''
    transform distance maps to depth maps.
    :param dist_maps: distance value maps from camera to poins
    :param cam_Ks: camera intrinsics
    :return: depth maps: z values from camera to points.
    '''

    depth_maps = np.ones_like(dist_maps) * np.inf

    view_id = 0
    for dist_map, cam_K in zip(dist_maps, cam_Ks):
        u, v = np.meshgrid(range(dist_map.shape[1]), range(dist_map.shape[0]))
        u = u.reshape([1, -1])[0]
        v = v.reshape([1, -1])[0]
        dist_cam = dist_map[v, u]

        non_inf_indices = np.argwhere(dist_cam<np.inf).T[0]
        dist_cam = dist_cam[non_inf_indices]
        u = u[non_inf_indices]
        v = v[non_inf_indices]

        # calculate coordinates
        x_temp = (u - cam_K[0][2]) / cam_K[0][0]
        y_temp = (v - cam_K[1][2]) / cam_K[1][1]
        z_temp = 1

        z = dist_cam / np.sqrt(x_temp**2 + y_temp**2 + z_temp**2)

        depth_maps[view_id, v, u] = z

        if 'erosion_size' in kwargs:
            # This is mainly result of experimenting.
            # The core idea is that the volume of the object is enlarged slightly
            # (by subtracting a constant from the depth map).
            # Dilation additionally enlarges thin structures (e.g. for chairs). (refers to Occupancy Network)
            depth_maps[view_id] = grey_erosion(depth_maps[view_id],
                                               size=(kwargs['erosion_size'], kwargs['erosion_size']))

        view_id += 1

    return depth_maps

def get_colors(values, palatte_name = 'RdBu_r', color_depth = 256):
    '''
    Return color values given scalars.
    :param values: N values
    :param palatte_name: check seaborn
    :param color_depth:
    :return: Nx3 colors
    '''
    import seaborn as sns
    palatte = np.array(sns.color_palette(palatte_name, color_depth))
    scaled_values = np.int32((values-min(values))/(max(values) - min(values)) * (color_depth-1))
    return palatte[scaled_values]

def pc_from_dep(depth_maps, cam_Ks, cam_RTs, rgb_imgs=None, store_camera=False):
    '''
    get point cloud from depth maps
    :param depth_maps: depth map list
    :param cam_Ks: corresponding camera intrinsics
    :param cam_RTs: corresponding camera rotations and translations
    :param rgb_imgs: corresponding rgb images of depth maps.
    :param store_camera: if calculate camera position and orientations.
    :return: aligned point clouds in the canonical system with color intensities.
    '''
    point_list_canonical = []
    camera_positions = []
    color_intensities = []

    if not isinstance(rgb_imgs, np.ndarray) and (rgb_imgs.shape[:3] == depth_maps.shape[:3]):
        store_color = False
        rgb_imgs = [[] for _ in range(len(depth_maps))]
    else:
        store_color = True

    for depth_map, rgb_img, cam_K, cam_RT in zip(depth_maps, rgb_imgs, cam_Ks, cam_RTs):
        u, v = np.meshgrid(range(depth_map.shape[1]), range(depth_map.shape[0]))
        u = u.reshape([1, -1])[0]
        v = v.reshape([1, -1])[0]

        z = depth_map[v, u]

        # remove infinitive pixels
        non_inf_indices = np.argwhere(z < np.inf).T[0]

        if isinstance(rgb_img, np.ndarray):
            color_indices = rgb_img[v, u][non_inf_indices]
        else:
            color_indices = []

        z = z[non_inf_indices]
        u = u[non_inf_indices]
        v = v[non_inf_indices]

        # calculate coordinates
        x = (u - cam_K[0][2]) * z / cam_K[0][0]
        y = (v - cam_K[1][2]) * z / cam_K[1][1]

        point_cam = np.vstack([x, y, z]).T

        point_canonical = (point_cam - cam_RT[:, -1]).dot(cam_RT[:, :-1])

        if store_camera:
            cam_pos = - cam_RT[:, -1].dot(cam_RT[:, :-1])
            focal_point = ([0, 0, 1] - cam_RT[:, -1]).dot(cam_RT[:, :-1])
            up = np.array([0, -1, 0]).dot(cam_RT[:, :-1])
            cam_pos = {'pos': cam_pos, 'fp': focal_point, 'up': up}
        else:
            cam_pos = {}

        point_list_canonical.append(point_canonical)
        camera_positions.append(cam_pos)
        color_intensities.append(color_indices)

    output = {'pc': point_list_canonical}
    if store_color:
        output['color'] = color_intensities

    if store_camera:
        output['cam'] = camera_positions

    return output

def estimate_point_normal(point_clouds):
    '''
    Estimate the normal vectors of each point in point cloud
    :param point_clouds (dict): point_clouds['pc']: point cloud data; point_clouds['cam']: camera settings.
    :return:
    '''
    n_points_per_view = [pcd.shape[0] for pcd in point_clouds['pc']]
    full_pcd = np.vstack(point_clouds['pc'])
    full_pcd_normal = pcu.estimate_point_cloud_normals(full_pcd, k=16)

    view_id = 0

    corrected_normals = []
    for cam, pc in zip(point_clouds['cam'], point_clouds['pc']):
        pc_normal = full_pcd_normal[sum(n_points_per_view[:view_id]):sum(n_points_per_view[:view_id + 1])]
        to_inverse_flags = np.matmul((pc - cam['pos'])[:, np.newaxis, :], pc_normal[:, :, np.newaxis]).flatten() > 0
        pc_normal[to_inverse_flags, :] *= -1
        corrected_normals.append(pc_normal)
        view_id += 1

    return corrected_normals

def normalize(a, axis=-1, order=2):
    '''
    Normalize any kinds of tensor data along a specific axis
    :param a: source tensor data.
    :param axis: data on this axis will be normalized.
    :param order: Norm order, L0, L1 or L2.
    :return:
    '''
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1

    if len(a.shape)==1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)

def multiview_render_pc_to_depth(points, cam_Ks, cam_RTs):
    '''
    Render the depth maps of pointcloud on multiple views.
    :param points (N x 3): point cloud.
    :param cam_Ks (N x 3 x 3): camera intrinsic parameters.
    :param cam_RTs (N x 3 x 3): camera extrinsic parameters.
    :return:
    '''
    n_views = cam_RTs.shape[0]
    n_points = points.shape[0]

    points = np.repeat(points[np.newaxis], n_views, axis=0)

    cam_pos = -np.matmul(cam_RTs[:, np.newaxis,:, -1], cam_RTs[:,:, :-1])
    focal_point = np.matmul(np.array([0, 0, 1])-cam_RTs[:,  np.newaxis, :,-1], cam_RTs[:, :, :-1])

    # camera system
    y_vec = normalize(np.matmul(np.repeat([[[0, 1, 0]]], n_views, axis = 0), cam_RTs[:, :, :-1]))
    z_vec = normalize(focal_point - cam_pos)
    x_vec = np.cross(y_vec, z_vec)

    Rot = np.transpose(np.concatenate([x_vec, y_vec, z_vec], axis=1), [0, 2, 1])

    # transform coordinates to camera system
    points_cam = np.transpose(np.matmul((points - np.repeat(cam_pos, n_points, axis=1)), Rot), [0, 2, 1])

    # get the depth value of each pixel location.
    pixels_loc = np.round(np.matmul(cam_Ks, points_cam / points_cam[:, -1, np.newaxis, :])[:, :-1]).astype(np.uint16)
    depths = points_cam[:, 2, :]

    depth_idx = np.repeat(np.argsort(depths, axis=-1)[:, np.newaxis], 2, axis=1)

    pixels_loc = np.take_along_axis(pixels_loc, depth_idx, axis=-1)
    depth_values = np.take_along_axis(depths, depth_idx[:,0], axis=-1)

    depth_maps = []
    for view_id in range(n_views):
        pixels_unique, pixels_index = np.unique(pixels_loc[view_id], axis=1, return_index=True)
        depth_unique = depth_values[view_id, pixels_index]

        depth_map = np.inf * np.ones([2 * int(cam_Ks[view_id, 1, 2]), 2 * int(cam_Ks[view_id, 0, 2])])
        depth_map[pixels_unique[1], pixels_unique[0]] = depth_unique

        depth_maps.append(depth_map)

    return depth_maps

def render_pc_to_depth(points, cam_K, cam_RT):
    '''
    Render the depth map of pointcloud given a view point.
    :param points (N x 3): point cloud.
    :param cam_K (3 x 3): camera intrinsic parameters.
    :param cam_RT (3 x 3): camera extrinsic parameters.
    :return:
    '''
    cam_pos = - cam_RT[:, -1].dot(cam_RT[:, :-1])
    focal_point = (np.array([0, 0, 1]) - cam_RT[:, -1]).dot(cam_RT[:, :-1])

    # camera system
    y_vec = normalize(np.array([0, 1, 0]).dot(cam_RT[:, :-1]))
    z_vec = normalize(focal_point - cam_pos)
    x_vec = np.cross(y_vec, z_vec)

    Rot = np.vstack([x_vec, y_vec, z_vec]).T

    # transform coordinates to camera system
    points_cam = ((points - cam_pos).dot(Rot)).T

    # get the depth value of each pixel location.
    pixels_loc = np.round(cam_K.dot(points_cam / points_cam[-1, :])[:-1, :]).astype(np.uint16)
    depths = points_cam[2, :]

    depth_idx = np.argsort(depths)
    pixels_loc = pixels_loc[:, depth_idx]
    depth_values = depths[depth_idx]

    pixels_unique, pixels_index = np.unique(pixels_loc, axis=1, return_index=True)
    depth_unique = depth_values[pixels_index]

    depth_map = np.inf * np.ones([2 * int(cam_K[1, 2]), 2 * int(cam_K[0, 2])])
    depth_map[pixels_unique[1], pixels_unique[0]] = depth_unique

    return depth_map

def farthest_point_sample(point, npoint):
    '''
    Sampling points with the specific number.
    :param pointcloud data (NxD):
    :param npoint: number of samples
    :return:
    centroids: sampled pointcloud index. (npoint x D)
    '''
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids.astype(np.int32)
