'''
This script runs the validation process for a specific dataset and scene, iterating through frames and executing a validation script for each frame.
'''

import sys
sys.path.append('.')

import os
from train_fcgsd import path_match

dataset_path = './data_video'
dataset = 'N3V'
scene_list = ['flame_steak']

for scene in scene_list:
    for frame in range(0, 300):
        ply_path = path_match(f'{dataset_path}/{dataset}/{scene}/frame{frame:06d}/gs/point_cloud/iteration_*/point_cloud.ply')
        source_path = f'{dataset_path}/{dataset}/{scene}/frame{frame:06d}'
        save_path = f'{dataset_path}/{dataset}/{scene}/frame{frame:06d}/gs'

        script = f'python validate.py --use_first_as_test --ply_path {ply_path} --source_path {source_path} --gpu 0 --save_path {save_path} '
        os.system(script)