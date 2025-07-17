import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
exp_name = 'test'
date = '0717'
lambda_size_list = [5e-3]

dataset_path = './data_video'



# # train
# dataset_train = 'meetroom'
# scene_list = ['trimming', 'vrheadset', 'stepin']
# for lambda_size in lambda_size_list:
#     model_path = f'./outputs/{date}-{lambda_size}-{exp_name}'
#     script = f"python train_dfcgs.py --gpu 0 --eval --dynamicGS_type control_point \
#         --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --motion_limit 2\
#         --frame_start 1 --frame_end 300 --gof_size 5 --iterations 4000\
#         --conduct_training  \
#         --scene_list {' '.join(scene_list)} --dataset {dataset_train} --dataset_path {dataset_path}\
#         --knn_num 30 --downsample_rate 70\
#         "
#     os.system(script)

# test
dataset_test = 'Immersive'
scene_list = ['04_Truck']
for lambda_size in lambda_size_list:
    model_path = f'./outputs/{date}-{lambda_size}-{exp_name}'
    checkpoint_path = f'ckpt/model.pth'
    script = f"python train_dfcgs.py --gpu 0 --eval --dynamicGS_type control_point\
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --motion_limit 0.05\
        --test_frame_start 0 --test_frame_end 9 --gof_size 10  --iterations 1000\
        --conduct_compress --conduct_decompress \
        --images images --model_path {model_path} --use_first_as_test \
        --scene_list {' '.join(scene_list)} --dataset {dataset_test} --dataset_path {dataset_path}\
        --checkpoint_path {checkpoint_path}\
        --knn_num 30 --downsample_rate 70\
        "
    os.system(script)

