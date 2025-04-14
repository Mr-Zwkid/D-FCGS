date = '0411'
import os
# lambda_size_list = [1e-3,  5e-3, 1e-4, 1e-5, 1e-2]
lambda_size_list = [5e-4, 5e-5, 5e-6]

scene_list = ['discussion']
for lambda_size in lambda_size_list:
    model_path = f'./3DGStream-Res/{date}-{lambda_size}-gof-exp0'
    # checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 0 --eval --use_scenes --use_gof  --dynamicGS_type control_point\
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1 --motion_limit 0.05\
        --frame_start 1 --frame_end 300 --test_frame_start 0 --test_frame_end 300 --gof_size 300  --iterations 1000\
        --conduct_compress --conduct_decompress \
        --images images --model_path {model_path} --use_first_as_test \
        --scene_list {' '.join(scene_list)} --dataset meetroom --knn_num 30 --downsample_rate 70\
        "
    os.system(script)