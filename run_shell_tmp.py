import os

scene_list = ['flame_steak-4']
lambda_size = 1e-2
for scene_name in scene_list:
    script = f"python train_fcgsd.py --gpu 3 --Q_y 1 --Q_z 1 --lambda_size {lambda_size}  \
        --conduct_compress --conduct_decompress --conduct_training --scene_name {scene_name} \
        --frame_start 1 --frame_end 21 --dynamicGS_type 3dgstream \
        --eval --images images_2 --iterations 300 --model_path ./3DGStream-Res/dynerf/{scene_name.split('-')[0]}-{lambda_size} --use_first_as_test"
    os.system(script)