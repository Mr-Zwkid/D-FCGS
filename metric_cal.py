# test PSNR/SSIM/LPIPS
import os
# dataset = 'meetroom'
# scene_list = ['discussion', 'stepin', 'vrheadset', 'trimming']
dataset = 'immersive'
scene_list = ['12_Cave']
for scene in scene_list:
    # for frame in range(1, 300):
    #     ply_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{dataset}/{scene}/frame{frame:06d}/point_cloud/iteration_150/point_cloud.ply'
    #     source_path = f'/SDD_D/zwk/data_dynamic/{dataset}/{scene}/frame{frame:06d}'
    #     save_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{dataset}/{scene}/frame{frame:06d}'

    #     script = f'python validate.py --use_first_as_test --ply_path {ply_path} --source_path {source_path} --gpu 1 --save_path {save_path} '
    #     os.system(script)
    for frame in range(0, 1):
        ply_path = '/SSD2/chenzx/Projects/Dataset4Compression/init_3dgs/immersive/12_Cave/frame000000/point_cloud/iteration_4000/point_cloud.ply'
        source_path = f'/SSD2/chenzx/Projects/Dataset4Compression/immersive/12_Cave/frame000000'
        save_path = f'/SSD2/chenzx/Projects/Dataset4Compression/init_3dgs/immersive/12_Cave/frame000000'

        script = f'python validate.py --use_first_as_test --ply_path {ply_path} --source_path {source_path} --gpu 1 --save_path {save_path} '
        os.system(script)