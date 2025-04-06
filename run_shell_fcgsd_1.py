# joint all scenes
import os
lambda_size_list = [1e-2]

scene_list = ['stepin', 'discussion', 'vrheadset']
for lambda_size in lambda_size_list:
    model_path = f'./3DGStream-Res/all_scenes-{lambda_size}-joint-stepping-0404-meetroom-gof5-e'
    checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 1 --eval  --dynamicGS_type 3dgstream_explicit \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1\
        --conduct_training --use_scenes --use_gof\
        --frame_start 1 --frame_end 300 --test_frame_start 1 --test_frame_end 10 --gof_size 5 \
        --images images --iterations 5000 --model_path {model_path} --use_first_as_test --dataset meetroom\
        --scene_list {' '.join(scene_list)} \
        "
    os.system(script)

scene_list = ['flame_steak']
for lambda_size in lambda_size_list:
    model_path = f'./3DGStream-Res/all_scenes-{lambda_size}-joint-stepping-0404-meetroom-gof5-e'
    checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 1 --eval --use_gof  --dynamicGS_type 3dgstream_explicit \
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1\
        --conduct_compress --conduct_decompress --use_scenes \
        --frame_start 1 --frame_end 30 --test_frame_start 1 --test_frame_end 30 --gof_size 10 \
        --images images --iterations 1000 --model_path {model_path} --use_first_as_test --dataset dynerf\
        --scene_list {' '.join(scene_list)} \
        "
    os.system(script)