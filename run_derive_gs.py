import os

scene_list = ['coffee_martini', 'cook_spinach', 'cut_roasted_beef', 'flame_salmon_1', 'sear_steak']

scripts = f"python derive_gs_frames_gof.py --dataset_dir ./data/N3V  --scene_list {' '.join(scene_list)} --start_frame 1 --end_frame 299 --gof_size 10 --first_load_iteration 4000\
 --load_iteration 150 --fcgs_lmd 0.0016 --fcgs_ckpt_dir /mnt/data3/ctx/FCGS/checkpoints --cuda_visible_devices 1"

os.system(scripts)