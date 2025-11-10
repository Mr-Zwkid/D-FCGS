import os

exp_name = 'test'
model_path = f'./outputs/{exp_name}' # save_path
checkpoint_path = f'ckpt/model.pth' # change this to your checkpoint path

dataset_path = './data_video'
dataset_name = 'Immersive'
scene_list = ['04_Truck']

script = f"python train_dfcgs.py --gpu 0 --eval --dynamicGS_type control_point --use_first_as_test\
    --conduct_compress --conduct_decompress \
    --test_frame_start 0 --test_frame_end 9 --gof_size 10\
    --model_path {model_path} --checkpoint_path {checkpoint_path}\
    --scene_list {' '.join(scene_list)} --dataset {dataset_name} --dataset_path {dataset_path}\
    --knn_num 30 --downsample_rate 100\
    --Q_y 1 --Q_z 1  --motion_limit 0.05\
    "
os.system(script)