import os
import json
import torch

torch.cuda.set_device(2)

dir_name = '/SDD_D/zwk/data_dynamic/dynerf'

for scene, threshold in zip(['flame_steak'], [31.5]):
    i = 2
    global_results = []
    global_cnt = 0
    while i < 5:
        os.system(f'python train.py  -m /SDD_D/zwk/init_3dgs/{scene}-{i}/frame000000 -s {os.path.join(dir_name, scene, f"frame000000")} --eval \
              --iterations 4000 --densify_until_iter 3000 --sh_degree 3' )
        os.system(f'python render.py  -m /SDD_D/zwk/init_3dgs/{scene}-{i}/frame000000 -s {os.path.join(dir_name, scene, "frame000000")}' )
        os.system(f'python metrics.py  -m /SDD_D/zwk//init_3dgs/{scene}-{i}/frame000000' )
        results_dir = f'/SDD_D/zwk/init_3dgs/{scene}-{i}/frame000000/results.json'
        with open(results_dir, 'r') as f:
            results = json.load(f)
        psnr = results["ours_7000"]["PSNR"]
        global_results.append(psnr)

        if psnr > threshold:
            i += 1
        global_cnt += 1
        if global_cnt >= 300:
            break
        print(scene, threshold)

    with open(f'./results/global_results_{scene}.txt', 'w') as f:
        for res in global_results:
            f.write(f'{res}\n')