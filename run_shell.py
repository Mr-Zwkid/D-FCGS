import os

lmbda_list = [4e-4, 2e-4, 1e-4]
exp_name = '3DGS_ori_mip360'
path_to_ply = '../gaussian-splatting/output/mipnerf360'
scene_list = os.listdir(path_to_ply)
for lmbda in lmbda_list:
    for scene in scene_list:
        print('-'*100,'\n', scene)
        os.system(f'python encode_single_scene.py --gpu 0 --lmd {lmbda} --ply_path_from {path_to_ply}/{scene}/point_cloud/iteration_30000/point_cloud.ply --bit_path_to ./outputs/{exp_name}/{scene} --determ 1')
        os.system(f'python decode_single_scene_validate.py --lmd {lmbda} --bit_path_from ./outputs/{exp_name}/{scene} --ply_path_to {path_to_ply}/{scene}/point_cloud/iteration_30000/point_cloud.ply --source_path ../data_static/mipnerf360/{scene}')
        