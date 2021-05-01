"""settings.py contains all configuration parameters the blender needs


reference: https://github.com/weiaicunzai/blender_shapenet_render
"""
import sys
sys.path.append('./')
from data_config import our_categories, shapenet_normalized_path, shapenet_categlory_pair, model_view_path

g_shapenet_path = shapenet_normalized_path
g_blender_excutable_path = '/home/ynie/Software/blender-2.79b-linux-glibc219-x86_64/blender'

#if you have multiple viewpoint files, add to the dict
#files contains azimuth,elevation,tilt angles and distance for each row
g_view_point_file = './view_points.txt'

g_render_objs = our_categories

#change this path to your background image folder
g_background_image_path = 'TRANSPARENT' # or 'background_image'

#folders to store synthetic data
g_syn_data_folder = './datasets/ShapeNetRenderings'
g_result_dict = model_view_path

#background image composite
#enum in [‘RELATIVE’, ‘ABSOLUTE’, ‘SCENE_SIZE’, ‘RENDER_SIZE’], default ‘RELATIVE’
g_scale_space = 'RENDER_SIZE'
g_use_film_transparent = True

#camera:
#enum in [‘QUATERNION’, ‘XYZ’, ‘XZY’, ‘YXZ’, ‘YZX’, ‘ZXY’, ‘ZYX’, ‘AXIS_ANGLE’]
g_rotation_mode = 'XYZ'

#output:
g_color_mode = 'RGB'
g_color_depth = '16'
g_file_format = 'OPEN_EXR'
g_use_file_extension = True

#enum in [‘BW’, ‘RGB’, ‘RGBA’], default ‘BW’
g_rgb_color_mode = 'RGB'
#enum in [‘8’, ‘10’, ‘12’, ‘16’, ‘32’], default ‘8’
g_rgb_color_depth = '16'
g_rgb_file_format = 'PNG'

g_depth_color_mode = 'RGB'
g_depth_color_depth = '16'
g_depth_file_format = 'OPEN_EXR'

g_depth_use_overwrite = True
g_depth_use_file_extension = True

#dimension:

#engine type [CYCLES, BLENDER_RENDER]
g_engine_type = 'CYCLES'

#output image size =  (g_resolution_x * resolution_percentage%, g_resolution_y * resolution_percentage%)
g_resolution_x = 512
g_resolution_y = 512
g_resolution_percentage = 100


#performance:
g_gpu_render_enable = False

#if you are using gpu render, recommand to set hilbert spiral to 256 or 512
#default value for cpu render is fine
g_hilbert_spiral = 512

#total 55 categories
g_shapenet_categlory_pair = shapenet_categlory_pair
cpu_cores = 4