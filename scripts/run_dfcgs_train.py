import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
exp_name = 'test'
date = '0724'
lambda_size_list = [5e-3]

dataset_path = './data_video'
dataset_list = ['Immersive', 'vru']
scene_list = ['04_Truck', '11_Alexa_Paint_2', 'b2']

for lambda_size in lambda_size_list:
    model_path = f'./outputs/{date}-{lambda_size}-{exp_name}'
    script = f"python train_dfcgs.py --gpu 0 --eval --dynamicGS_type control_point \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --motion_limit 2\
        --frame_start 0 --frame_end 99 --gof_size 5 --iterations 4000\
        --conduct_training  \
        --scene_list {' '.join(scene_list)} --dataset_list {' '.join(dataset_list)} --dataset_path {dataset_path}\
        --knn_num 30 --downsample_rate 70\
        --model_path {model_path}\
        "
    os.system(script)

