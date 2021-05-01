""" render_rgb.py renders obj file to rgb image

Aviable function:
- clear_mash: delete all the mesh in the secene
- scene_setting_init: set scene configurations
- node_setting_init: set node configurations
- render: render rgb image for one obj file and one viewpoint
- render_obj_by_vp_lists: wrapper function for render() render
                          one obj file by multiple viewpoints
- render_objs_by_one_vp: wrapper function for render() render
                         multiple obj file by one viewpoint
- init_all: a wrapper function, initialize all configurations
= set_image_path: reset defualt image output folder

author baiyu
"""
import sys
import os
import pickle
import numpy as np
import bpy
from mathutils import Matrix
import argparse

abs_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(abs_path))

from render_helper import *
from settings import *
from data_config import camera_setting_path, total_view_nums

def clear_mesh():
    """ clear all meshes in the secene

    """
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)

    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)

    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)

    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH' or obj.type == 'EMPTY':
            obj.select = True
    bpy.ops.object.delete()

def scene_setting_init(use_gpu):
    """initialize blender setting configurations

    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = g_engine_type
    bpy.data.scenes[sce].cycles.film_transparent = g_use_film_transparent
    #output
    bpy.data.scenes[sce].render.image_settings.color_mode = g_color_mode
    bpy.data.scenes[sce].render.image_settings.color_depth = g_color_depth
    bpy.data.scenes[sce].render.image_settings.file_format = g_file_format
    bpy.data.scenes[sce].render.use_file_extension = g_use_file_extension

    #dimensions
    bpy.data.scenes[sce].render.resolution_x = g_resolution_x
    bpy.data.scenes[sce].render.resolution_y = g_resolution_y
    bpy.data.scenes[sce].render.resolution_percentage = g_resolution_percentage

    if use_gpu:
        bpy.data.scenes[sce].render.engine = 'CYCLES' #only cycles engine can use gpu
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        bpy.data.scenes[sce].render.tile_x = g_hilbert_spiral
        cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
        for device in cycles_prefs.devices:
            if device.type == 'CUDA':
                device.use = True
        bpy.context.user_preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        bpy.data.scenes[sce].cycles.device = 'GPU'

def node_setting_init():
    """node settings for render rgb images

    mainly for compositing the background images
    """


    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    img_file_output_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_file_output_node = tree.nodes.new("CompositorNodeOutputFile")

    scale_node.space = g_scale_space

    img_file_output_node.format.color_mode = g_rgb_color_mode
    img_file_output_node.format.color_depth = g_rgb_color_depth
    img_file_output_node.format.file_format = g_rgb_file_format
    img_file_output_node.base_path = g_syn_data_folder

    depth_file_output_node.format.color_mode = g_depth_color_mode
    depth_file_output_node.format.color_depth = g_depth_color_depth
    depth_file_output_node.format.file_format = g_depth_file_format
    depth_file_output_node.base_path = g_syn_data_folder

    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
    links.new(alpha_over_node.outputs[0], img_file_output_node.inputs[0])
    links.new(render_layer_node.outputs['Depth'], depth_file_output_node.inputs[0])

def render(viewpoint, viewpoint_id, rendering_dir):
    """render rgb image and depth maps

    render a object rgb image by a given camera viewpoint and
    choose random image as background, only render one image
    at a time.

    Args:
        viewpoint: a vp parameter(contains azimuth,elevation,tilt angles and distance)
        viewpoint_id: the index of viewpoint
        rendering_dir: path to store camera info
    """

    vp = viewpoint
    cam_location = camera_location(vp.azimuth, vp.elevation, vp.distance)
    cam_rot = camera_rot_XYZEuler(vp.azimuth, vp.elevation, vp.tilt)

    cam_obj = bpy.data.objects['Camera']
    cam_obj.location[0] = cam_location[0]
    cam_obj.location[1] = cam_location[1]
    cam_obj.location[2] = cam_location[2]

    cam_obj.rotation_euler[0] = cam_rot[0]
    cam_obj.rotation_euler[1] = cam_rot[1]
    cam_obj.rotation_euler[2] = cam_rot[2]

    if g_background_image_path == 'TRANSPARENT':
        bpy.context.scene.render.alpha_mode = g_background_image_path
    else:
        background_images = os.listdir(g_background_image_path)
        image_name = random.choice(background_images)
        image_path = os.path.join(g_background_image_path, image_name)
        image_node = bpy.context.scene.node_tree.nodes[0]
        image_node.image = bpy.data.images.load(image_path)

    img_file_output_node = bpy.context.scene.node_tree.nodes[4]
    img_file_output_node.file_slots[0].path = 'color_###.png' # blender placeholder #

    depth_file_output_node = bpy.context.scene.node_tree.nodes[5]
    depth_file_output_node.file_slots[0].path = 'depth_###.exr' # blender placeholder #

    #start rendering
    bpy.context.scene.frame_set(viewpoint_id + 1)
    bpy.ops.render.render(write_still=True)

    # write camera info
    cam_K_file = os.path.join(cam_K_path, 'cam_K.txt')
    if (not os.path.isfile(cam_K_file)) or (len(os.listdir(cam_RT_path))<total_view_nums):
        K, RT = get_3x4_P_matrix_from_blender(cam_obj)
        np.savetxt(cam_K_file, K)
        np.savetxt(os.path.join(cam_RT_path, 'cam_RT_{0:03d}.txt'.format(viewpoint_id + 1)), RT)
        print('Camera parameters written.')

def render_obj_by_vp_lists(rendering_dir, viewpoints):
    """ render one obj file by a given viewpoint list
    a wrapper function for render()

    Args:
        rendering_dir: a string variable indicate the rendering path of the model.
        viewpoints: an iterable object of vp parameter(contains azimuth,elevation,tilt angles and distance)
    """

    if isinstance(viewpoints, tuple):
        vp_lists = [viewpoints]

    try:
        vp_lists = iter(viewpoints)
    except TypeError:
        print("viewpoints is not an iterable object")

    for vp_id, vp in enumerate(vp_lists):
        set_image_path(rendering_dir)
        set_depth_path(rendering_dir)
        render(vp, vp_id, rendering_dir)

def render_objs_by_one_vp(obj_pathes, viewpoint):
    """ render multiple obj files by a given viewpoint

    Args:
        obj_paths: an iterable object contains multiple
                   obj file pathes
        viewpoint: a namedtuple object contains azimuth,
                   elevation,tilt angles and distance
    """

    if isinstance(obj_pathes, str):
        obj_lists = [obj_pathes]

    try:
        obj_lists = iter(obj_lists)
    except TypeError:
        print("obj_pathes is not an iterable object")

    for obj_path in obj_lists:
        rendering_dir = os.path.join(output_folder, obj_path.split('/')[4])
        if not os.path.exists(rendering_dir):
            os.makedirs(rendering_dir)

        clear_mesh()
        bpy.ops.import_scene.obj(filepath=obj_path)
        set_image_path(rendering_dir)
        set_depth_path(rendering_dir)
        render(viewpoint, 1, rendering_dir)

def camera_setting_init():
    """ camera settings for renderer
    """
    bpy.data.objects['Camera'].rotation_mode = g_rotation_mode

def light_setting_init():
    """ light settings for renderer
    """
    # Make light just directional, disable shadows.
    world = bpy.data.worlds['World']
    world.use_nodes = True

    # changing these values does affect the render.
    bg = world.node_tree.nodes['Background']
    bg.inputs[1].default_value = 10.0

def init_all():
    """init everything we need for rendering
    an image
    """
    scene_setting_init(g_gpu_render_enable)
    camera_setting_init()
    node_setting_init()
    light_setting_init()

def set_image_path(new_path):
    """ set image output path to new_path

    Args:
        new rendered image output path
    """
    file_output_node = bpy.context.scene.node_tree.nodes[4]
    file_output_node.base_path = new_path

def set_depth_path(new_path):
    """ set image output path to new_path

    Args:
        new rendered depth output path
    """
    file_output_node = bpy.context.scene.node_tree.nodes[5]
    file_output_node.base_path = new_path

#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_blender2shapenet = Matrix(
        ((1, 0, 0),
         (0, 0, -1),
         (0, 1, 0)))

    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam*R_blender2shapenet
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K, RT

### YOU CAN WRITE YOUR OWN IMPLEMENTATION TO GENERATE DATA
def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    parser = argparse.ArgumentParser(description='Blender renderer.')
    parser.add_argument("dict", type=str,
                        help="model-view file for rendering.")
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':

    args = parse_args()

    init_all()

    result_list = pickle.load(open(args.dict, 'rb'))

    cam_K_path = os.path.join(camera_setting_path, 'cam_K')
    cam_RT_path = os.path.join(camera_setting_path, 'cam_RT')
    if not os.path.exists(cam_K_path):
        os.makedirs(cam_K_path)
    if not os.path.exists(cam_RT_path):
        os.makedirs(cam_RT_path)

    for model in result_list:
        cat = model.path.split('/')[3]
        output_folder = os.path.join(g_syn_data_folder, cat)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        rendering_dir = os.path.join(output_folder, model.path.split('/')[4])
        if not os.path.exists(rendering_dir):
            os.makedirs(rendering_dir)
        if len(os.listdir(rendering_dir)) == 40:
            print('Rendering has been done with this model.')
            continue
        clear_mesh()
        bpy.ops.import_scene.obj(filepath=model.path)
        render_obj_by_vp_lists(rendering_dir, model.vps)
