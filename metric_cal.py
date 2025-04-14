# test PSNR/SSIM/LPIPS
import os
dataset = 'meetroom'
scene_list = ['discussion', 'stepin', 'vrheadset', 'trimming']
for scene in scene_list:
    for frame in range(1, 300):
        ply_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{dataset}/{scene}/frame{frame:06d}/point_cloud/iteration_150/point_cloud.ply'
        source_path = f'/SDD_D/zwk/data_dynamic/{dataset}/{scene}/frame{frame:06d}'
        save_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{dataset}/{scene}/frame{frame:06d}'

        script = f'python validate.py --use_first_as_test --ply_path {ply_path} --source_path {source_path} --gpu 1 --save_path {save_path} '
        os.system(script)