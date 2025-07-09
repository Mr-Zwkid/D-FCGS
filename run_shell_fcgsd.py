date = '0614'
import os
lambda_size_list = [5e-3]
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
scene_list = os.listdir('./output_gt/meetroom')
scene_list = ['trimming', 'vrheadset', 'stepin']
exp_name = 'one_stage'

dataset_train = 'meetroom'
# train
for lambda_size in lambda_size_list:
    model_path = f'./rebuttal/{date}-{lambda_size}-{exp_name}'
    checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 2 --eval --use_scenes --use_gof  --dynamicGS_type control_point \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --motion_limit 2\
        --frame_start 1 --frame_end 300 --gof_size 5 --iterations 4000\
        --conduct_training  \
        --images images  --model_path {model_path} --use_first_as_test \
        --scene_list {' '.join(scene_list)} --dataset {dataset_train} --knn_num 30 --downsample_rate 70\
        "
    os.system(script)

# # test
# scene_list = ['cut_roasted_beef']
# dataset_test = 'dynerf'
# for lambda_size in lambda_size_list:
#     model_path = f'./rebuttal/{date}-{lambda_size}-{exp_name}'
#     # checkpoint_path = f'{model_path}/.pth'
#     script = f"python train_fcgsd.py --gpu 3 --eval --use_scenes --use_gof  --dynamicGS_type control_point\
#         --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --motion_limit 0.05\
#         --test_frame_start 1 --test_frame_end 100 --gof_size 100  --iterations 1000\
#         --conduct_compress --conduct_decompress \
#         --images images --model_path {model_path} --use_first_as_test \
#         --scene_list {' '.join(scene_list)} --dataset {dataset_test} --knn_num 30 --downsample_rate 70\
#         "
#     os.system(script)

