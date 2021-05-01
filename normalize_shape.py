'''
Normalize shapenet obj files to [-0.5, 0.5]^3.

author: ynie
date: Jan, 2020
'''
import sys
sys.path.append('.')
from data_config import shape_scale_padding, \
    shapenet_path, shapenet_normalized_path
import os
from multiprocessing import Pool
from tools.utils import append_dir, normalize_obj_file
from tools.read_and_write import write_json, load_data_path
from settings import cpu_cores

def recursive_normalize(input_path, output_path):
    '''
    Normalize *.obj file recursively
    :param input_path:
    :param output_path:
    :return:
    '''
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    for root, _, files in os.walk(input_path, topdown=True):
        for file in files:
            input_file_path = os.path.join(root, file)
            output_file_path = input_file_path.replace(input_path, output_path)

            if not os.path.exists(os.path.dirname(output_file_path)):
                os.makedirs(os.path.dirname(output_file_path))

            if not file.endswith('.obj'):
                if os.path.exists(output_file_path):
                    os.remove(output_file_path)
                os.symlink(input_file_path, output_file_path)
                continue
            else:
                # write obj file
                size_centroid_file = '.'.join(output_file_path.split('.')[:-1]) + '_size_centroid.json'

                if os.path.exists(output_file_path) and os.path.exists(size_centroid_file):
                    continue

                total_size, centroid = normalize_obj_file(input_file_path, output_file_path,
                                                          padding=shape_scale_padding)

                size_centroid = {'size': total_size.tolist(), 'centroid': centroid.tolist()}

                write_json(size_centroid_file, size_centroid)

def normalize(obj_path):
    '''
    normalize shapes
    :param obj_path: ShapeNet object path
    :return:
    '''
    cat, obj_file = obj_path.split('/')[3:5]

    input_cat_dir = append_dir(os.path.join(shapenet_path, cat), obj_file, 'i')
    output_cat_dir = append_dir(os.path.join(shapenet_normalized_path, cat), obj_file, 'o')

    recursive_normalize(input_cat_dir, output_cat_dir)

if __name__ == '__main__':
    if not os.path.exists(shapenet_normalized_path):
        os.mkdir(shapenet_normalized_path)

    all_objects = load_data_path(shapenet_path)

    p = Pool(processes=cpu_cores)
    p.map(normalize, all_objects)
    p.close()
    p.join()