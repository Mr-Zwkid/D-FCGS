import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
exp_name = 'train-meetroom-1'
date = '1223'
lambda_size_list = [2e-3]

dataset_path = '/mnt/data3/ctx/D-FCGS_old/data'

dataset_list = ['Meetroom']
# # scene_list = ['stepin', 'trimming', 'vrheadset']
scene_list = ['trimming']

# dataset_list = ['Sunny']
# scene_list = sorted(os.listdir(os.path.join(dataset_path, 'Sunny')))[:30]

# scene_list = [s for s in scene_list if os.path.getsize(os.path.join(dataset_path, 'Sunny', s, 'frame000000/gs/point_cloud/iteration_4000/point_cloud.ply')) / 1024 / 1024 < 100 and os.path.exists(os.path.join(dataset_path, 'Sunny', s, 'frame000099/gs/point_cloud/iteration_150/point_cloud.ply'))]
# print(len(scene_list))
# print('scene_list:', scene_list)


for lambda_size in lambda_size_list:
    model_path = f'./outputs/{date}-{lambda_size}-{exp_name}'
    script = f"python train_fcgsd.py --gpu 0 --eval --dynamicGS_type control_point \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --motion_limit 2\
        --frame_start 0 --frame_end 99 --gof_size 10 --iterations 1000\
        --conduct_training  \
        --scene_list {' '.join(scene_list)} --dataset_list {' '.join(dataset_list)} --dataset_path {dataset_path} --use_scenes\
        --knn_num 30 --downsample_rate 16 --checkpoint_path ./ckpt/model.pth\
        --model_path {model_path}\
        "
    os.system(script)

