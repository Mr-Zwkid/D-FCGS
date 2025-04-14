date = '0411'
import os
lambda_size_list = [ 5e-4]
# # lambda_size_list = [ 5e-3]
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# # scene_list = os.listdir('./output_gt/meetroom')
# scene_list = ['trimming', 'vrheadset', 'stepin']
# for lambda_size in lambda_size_list:
#     model_path = f'./3DGStream-Res/{date}-{lambda_size}-gof-implicit'
#     checkpoint_path = f'{model_path}/.pth'
#     script = f"python train_fcgsd.py --gpu 3 --eval --use_scenes --use_gof  --dynamicGS_type 3dgstream_implicit \
#         --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1 --motion_limit 2\
#         --frame_start 1 --frame_end 300 --test_frame_start 20 --test_frame_end 300 --gof_size 5 --iterations 3000\
#         --conduct_training  \
#         --images images  --model_path {model_path} --use_first_as_test \
#         --scene_list {' '.join(scene_list)} --dataset meetroom --knn_num 30 --downsample_rate 70\
#         "
#     os.system(script)

# # samlpe test
# scene_list = ['cut_roasted_beef']
scene_list = ['coffee_martini', 'flame_steak']
# scene_list = ['sear_steak-5', 'cut_roasted_beef']
for lambda_size in lambda_size_list:
    model_path = f'./3DGStream-Res/{date}-{lambda_size}-gof-implicit'
    # checkpoint_path = f'{model_path}/.pth'
    script = f"python train_fcgsd.py --gpu 3 --eval --use_scenes --use_gof  --dynamicGS_type 3dgstream_implicit\
        --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  --residual_num 1 --motion_limit 0.05\
        --frame_start 1 --frame_end 300 --test_frame_start 1 --test_frame_end 300 --gof_size 300  --iterations 1000\
        --conduct_compress --conduct_decompress \
        --images images --model_path {model_path} --use_first_as_test \
        --scene_list {' '.join(scene_list)} --dataset dynerf --knn_num 30 --downsample_rate 70\
        "
    os.system(script)

