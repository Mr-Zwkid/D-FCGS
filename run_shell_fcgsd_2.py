# scene_list = ['sear_steak-5', 'flame_salmon_1-3', 'cut_roasted_beef-8', 'coffee_martini-4', 'flame_steak-4', 'cook_spinach-3']
# joint exp
import os
scene_list = ['cook_spinach-3']
# scene_list = ['cut_roasted_beef-8']
# lambda_size = 1e-2
for lambda_size in [1e-2, 1e-3, 1e-4]:
    for scene_name in scene_list:
        model_path = f'./3DGStream-Res/dynerf/{scene_name.split("-")[0]}-{lambda_size}-joint'
        checkpoint_path = f'{model_path}/model.pth'
        script = f"python train_fcgsd.py --gpu 2 --eval --joint  --dynamicGS_type 3dgstream \
            --Q_y 1 --Q_z 1 --lr 1e-3 --lambda_size {lambda_size}  \
            --conduct_compress --conduct_decompress --conduct_train --scene_name {scene_name} \
            --frame_start 1 --frame_end 300 --test_frame_start 1 --test_frame_end 300\
            --images images_2 --iterations 1200 --model_path {model_path} --checkpoint 100 --use_first_as_test"
        os.system(script)