

import os

exp_name = 'test_final_model'
model_path = f'./outputs/{exp_name}' # save_path
checkpoint_path = f'ckpt/model.pth' # change this to your checkpoint path


dataset_path = './data'
dataset_name = 'N3V'
scene_list = ['coffee_martini']

script = f"python train_fcgsd.py --gpu 0 --eval --dynamicGS_type control_point --use_first_as_test\
    --conduct_compress --conduct_decompress \
    --test_frame_start 0 --test_frame_end 9 --gof_size 10\
    --model_path {model_path} --checkpoint_path {checkpoint_path}\
    --scene_list {' '.join(scene_list)} --dataset_list {dataset_name} --dataset_path {dataset_path}\
    --knn_num 30 --downsample_rate 100\
    --Q_y 1 --Q_z 1  --motion_limit 2 \
    --lmd 16e-4 \
    --use_I_compress\
    --motion_xyz_quant int8_dz --motion_xyz_quant_step 5e-3 --motion_xyz_deadzone 0.0015 --motion_xyz_quant_bias mean \
    "
os.system(script)

# --use_added \
# --use_I_compress \
# --motion_xyz_stats