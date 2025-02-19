import os

# scene_list = ['sear_steak-5', 'flame_salmon_1-3', 'cut_roasted_beef-8', 'coffee_martini-4', 'flame_steak-4', 'cook_spinach-3']
scene_list = ['coffee_martini-4']

for scene_name in scene_list:
    script = f"python train_fcgsd.py --scene_name {scene_name} --frame_start 1 --frame_end 5 --dynamicGS_type 3dgstream \
        --eval --images images_2 --iterations 10 --model_path ./3DGStream-Res/dynerf/{scene_name.split('-')[0]}"
    os.system(script)