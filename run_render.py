"""Parallel rendering script

Author: Yinyu
Reference:
https://github.com/weiaicunzai/blender_shapenet_render
"""

import subprocess
import pickle
from time import time
from render_helper import *
from settings import *
from data_config import total_view_nums, shapenet_normalized_path
import multiprocessing
from tools.read_and_write import load_data_path

def render_cmd(result_dict):
    #render rgb
    command = [g_blender_excutable_path, '--background', '--python', 'render_all.py', '--', result_dict]
    subprocess.run(command)

def group_by_cpu(result_list, count):
    '''
    deploy model-view result to different kernels
    :param result_list: model-view result
    :param count: kernel count
    :return:
    '''
    num_per_batch = int(len(result_list)/count)
    result_list_by_group = []
    for batch_id in range(count):
        if batch_id != count-1:
            result_list_by_group.append(result_list[batch_id*num_per_batch: (batch_id+1)*num_per_batch])
        else:
            result_list_by_group.append(result_list[batch_id * num_per_batch:])
    return result_list_by_group

if __name__ == '__main__':
    '''sample viewpoints'''
    all_objects = load_data_path(shapenet_normalized_path)
    result_list = collect_obj_and_vps(all_objects, total_view_nums)

    model_view_dir = os.path.dirname(g_result_dict)
    if not os.path.exists(model_view_dir):
        os.mkdir(model_view_dir)

    # save for parallel rendering
    count = 4
    print('Core Num: %d are used.' % (count))

    result_list_by_group = group_by_cpu(result_list, count)

    model_view_paths = []
    for batch_id, result_per_group in zip(range(count), result_list_by_group):
        file_name = str(batch_id) + '_' + os.path.basename(g_result_dict)
        single_model_view_path = os.path.join(model_view_dir, file_name)
        model_view_paths.append(single_model_view_path)

        with open(single_model_view_path, 'wb') as f:
            pickle.dump(result_per_group, f)

    pool = multiprocessing.Pool(processes=count)
    start = time()
    pool.map(render_cmd, model_view_paths)
    end = time()
    print('Time elapsed: %f.' % (end-start))