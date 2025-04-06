# joint all scenes
import os
lambda_size_list = [1e-3]
scene_list = os.listdir('./output_gt/cmu')
for lambda_size in lambda_size_list:
    model_path = f'./3DGStream-Res/all_scenes-{lambda_size}-joint-stepping-0406-cmu-tmp'
    checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 3 --eval --use_scenes --use_gof  --dynamicGS_type 3dgstream_implicit \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1 --motion_limit 1\
        --frame_start 5 --frame_end 30 --test_frame_start 20 --test_frame_end 300 --gof_size 5 --iterations 1000\
        --conduct_training  \
        --images images  --model_path {model_path} --use_first_as_test \
        --scene_list {' '.join(scene_list)} --dataset cmu\
        "
    os.system(script)

# joint all scenes
scene_list = ['flame_steak']
for lambda_size in lambda_size_list:
    model_path = f'./3DGStream-Res/all_scenes-{lambda_size}-joint-stepping-0406-cmu-tmp'
    checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 3 --eval --use_gof  --dynamicGS_type 3dgstream_implicit \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1 --motion_limit 0.05\
        --conduct_compress --conduct_decompress --use_scenes \
        --frame_start 1 --frame_end 300 --test_frame_start 0 --test_frame_end 20 --gof_size 20 \
        --images images --iterations 1000 --model_path {model_path} --use_first_as_test --dataset dynerf\
        --scene_list {' '.join(scene_list)} \
        "
    os.system(script)