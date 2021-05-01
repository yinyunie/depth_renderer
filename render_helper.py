"""render_helper.py contains functions that processing
data to the format we want


Available functions:
- load_viewpoint: read viewpoint file, load viewpoints
- load_viewpoints: wrapper function for load_viewpoint
- load_object_lists: return a generator of object file pathes
- camera_location: return a tuple contains camera location (x, y, z)
    in world coordinates system
- camera_rot_XYZEuler: return a tuple contains cmera ration
- random_sample_objs: sample obj files randomly
- random_sample_vps: sample vps from viewpoint file
- random_sample_objs_and_vps: wrapper function for sample obj
                              and viewpoints
author baiyu
"""

import os
import glob
import math
import random

from collections import namedtuple
from settings import *

# need to write outside the function, otherwise pickle can find
# where VP were defined
VP = namedtuple('VP', ['azimuth', 'elevation', 'tilt', 'distance'])
Model = namedtuple('Model', ['path', 'vps'])


def load_viewpoint(viewpoint_file):
    """read viewpoints from a file, can only read one file at once

    Args:
        viewpoint_file: file path to viewpoint file, read only one file
        for each function call

    Returns:
        generator of viewpoint parameters(contains azimuth,elevation,tilt angles and distance)
    """
    with open(viewpoint_file) as viewpoints:
        for line in viewpoints.readlines():
            yield VP(*line.strip().split())


def load_viewpoints(viewpoint_file_list):
    """load multiple viewpoints file from given lists

    Args:
        viewpoint_file_list: a list contains obj path
        a wrapper for load_viewpoint function

    Returns:
        return a generator contains multiple generators
        which contains obj pathes
    """

    if isinstance(viewpoint_file_list, str):
        vp_file_list = [viewpoint_file_list]

    try:
        vp_file_list = iter(viewpoint_file_list)
    except TypeError:
        print("viewpoint_file_list is not an iterable object")

    for vp_file in vp_file_list:
        yield load_viewpoint(vp_file)


def load_object_lists(category=None):
    """
        load object pathes according to the given category

    Args:
        category:a iterable object contains the category which
            we want render

    Returns:
        generator of gnerators of obj file pathes
    """

    # type checking

    # not empty
    assert category

    # not string
    if isinstance(category, str):
        category = [category]

    # iterable
    try:
        iter(category)
    except TypeError:
        print("category should be an iterable object")

    # a subset of the full category set
    assert set(category).issubset(g_shapenet_categlory_pair.values())

    # load obj file path
    for cat in category:
        search_path = os.path.join(g_shapenet_path, cat, '**', '*.obj')
        yield glob.iglob(search_path, recursive=True)


def camera_location(azimuth, elevation, dist):
    """get camera_location (x, y, z)

    you can write your own version of camera_location function
    to return the camera loation in the blender world coordinates
    system

    Args:
        azimuth: azimuth radius(object centered)
        elevation: elevation radius(object centered)
        dist: distance between camera and object(in meter)

    Returens:
        return the camera location in world coordinates in meters
    """

    phi = float(elevation)
    theta = float(azimuth)
    dist = float(dist)

    x = dist * math.cos(phi) * math.cos(theta)
    y = dist * math.cos(phi) * math.sin(theta)
    z = dist * math.sin(phi)

    return x, y, z


def camera_rot_XYZEuler(azimuth, elevation, tilt):
    """get camera rotaion in XYZEuler

    Args:
        azimuth: azimuth radius(object centerd)
        elevation: elevation radius(object centerd)
        tilt: twist radius(object centerd)

    Returns:
        return the camera rotation in Euler angles(XYZ ordered) in radians
    """

    azimuth, elevation, tilt = float(azimuth), float(elevation), float(tilt)
    x, y, z = math.pi/2, 0, math.pi/2  # set camera at x axis facing towards object

    # twist
    # if tilt > 0:
    #    y = tilt
    # else:
    #    y = 360 + tilt

    # latitude
    x = x - elevation
    # longtitude
    z = z + azimuth

    return x, y, z

def random_sample_objs(num_per_cat):
    """randomly sample object file from ShapeNet for each
    category in global variable g_render_objs, and then
    save the result in global variable g_obj_path

    Args:
        num_per_cat: how many obj file we want to sample per
        category

    Returns:
        vps: a list contains obj file path
    """

    obj_path_lists = load_object_lists(g_render_objs)
    obj_path_list = []

    for pathes in obj_path_lists:
        pathes = list(pathes)
        random.shuffle(pathes)
        if num_per_cat > len(pathes):
            num_per_cat = len(pathes)
        samples = random.sample(pathes, num_per_cat)
        obj_path_list += samples

    return obj_path_list

def random_sample_vps(obj_pathes, num_per_model):
    """randomly sample vps from vp lists, for each model,
    we sample num_per_cat number vps, and save the result to
    g_vps
    Args:
        obj_pathes: result of function random_sample_objs,
                    contains obj file pathes
        num_per_cat: how many view point to sample per model

    Returns:
        result_list: a list contains model name and its corresponding
             viewpoints
    """
    viewpoint_list = list(load_viewpoint(g_view_point_file))

    if num_per_model >= len(viewpoint_list):
        viewpoint_samples = viewpoint_list
    else:
        viewpoint_samples = random.sample(viewpoint_list, num_per_model)

    result_list = [Model(path, viewpoint_samples) for path in obj_pathes]
    return result_list

def random_sample_objs_and_vps(model_num_per_cat, vp_num_per_model):
    """wrapper function for randomly sample model and viewpoints
    and return the result, each category in g_render_objs contains
    multiple Model object, each Model object has path and vps attribute
    path attribute indicates where the obj file is and vps contains
    viewpoints to render the obj file

    Args:
        model_num_per_cat: how many models you want to sample per category
        vp_num_per_model: how many viewpoints you want to sample per model

    Returns:
        return a list contains Model objects
    """

    obj_pathes = random_sample_objs(model_num_per_cat)
    result_list = random_sample_vps(obj_pathes, vp_num_per_model)

    return result_list

def collect_obj_and_vps(all_objects, vp_num_per_model):
    """wrapper function for randomly sample viewpoints for each model you want to render
    and return the result, each category in g_render_objs contains
    multiple Model object, each Model object has path and vps attribute
    path attribute indicates where the obj file is and vps contains
    viewpoints to render the obj file

    Args:
        all_objects: all models you want to render
        vp_num_per_model: how many viewpoints you want to sample per model

    Returns:
        return a list contains Model objects
    """
    result_list = random_sample_vps(all_objects, vp_num_per_model)
    return result_list
