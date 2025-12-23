import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize quality metrics (PSNR, SSIM, etc.)")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the results')
    parser.add_argument('--dataset_path', type=str, default='./data_video', help='Path to the dataset (e.g., immersive, vru)')
    parser.add_argument('--dataset', type=str, default='immersive', help='Dataset name (e.g., immersive, vru)')
    parser.add_argument('--scene_list', nargs='+', required=True, help='List of scenes to analyze')
    parser.add_argument('--start_frame', type=int, default=0, help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=99, help='End frame index (inclusive)')
    parser.add_argument('--gof_size', type=int, default=10, help='Group of frames size for analysis')
    args = parser.parse_args()

    base_dir = args.base_dir
    dataset_path = args.dataset_path
    dataset = args.dataset
    scene_list = args.scene_list
    frame_start = args.start_frame
    frame_end = args.end_frame
    gof_size = args.gof_size

    scene_averages = []

    for scene in scene_list:
        base_dir_cur = f'{base_dir}/{scene}'
        gt_dir = f'{dataset_path}/{dataset}/{scene}'

        psnr_list = []
        ssim_list = []
        lpips_list = []
        
        psnr_gt_list = []
        ssim_gt_list = []
        lpips_gt_list = []
        
        psnr_diff_list = []
        ssim_diff_list = []
        lpips_diff_list = []
        
        frame_list = []

        for frame in range(frame_start, frame_end + 1):
            print('Processing frame:', frame)

            try:
                gt_path_psnr = f'{gt_dir}/frame{frame:06d}/gs/rendering_info.json'
                with open(gt_path_psnr, 'r') as f:
                    gt = json.load(f)
                    psnr_gt = gt['average']['PSNR']
                    psnr_gt_list.append(psnr_gt)
                    ssim_gt = gt['average']['SSIM']
                    ssim_gt_list.append(ssim_gt)
                    lpips_gt = gt['average'].get('LPIPS', None)
                    lpips_gt_list.append(lpips_gt)
            except Exception as e:
                print(f"Error reading {gt_path_psnr}: {e}")

                print('Using alternative ground truth data source from 3DGStream.')

                gt_path_psnr = f'{gt_dir}/frame{frame:06d}/gs/results.json'
                with open(gt_path_psnr, 'r') as f:
                    gt = json.load(f)
                    psnr_gt = gt['stage1/psnr_0']
                    psnr_gt_list.append(psnr_gt)
                    ssim_gt = None
                    ssim_gt_list.append(ssim_gt)
                    lpips_gt = None
                    lpips_gt_list.append(lpips_gt)

            frame_list.append(frame)

            if frame % gof_size == 0:
                psnr_list.append(psnr_gt)
                ssim_list.append(ssim_gt)
                lpips_list.append(lpips_gt)
                psnr_diff_list.append(0)
                ssim_diff_list.append(0)
                lpips_diff_list.append(0)
                continue

            path_psnr = f'{base_dir_cur}/frame{frame:06d}/rendering_info.json'
            with open(path_psnr, 'r') as f:
                cur = json.load(f)
                psnr = cur['average']['PSNR']
                psnr_list.append(psnr)
                psnr_diff = psnr - psnr_gt if psnr_gt is not None else None
                psnr_diff_list.append(psnr_diff)

                ssim = cur['average']['SSIM']
                ssim_list.append(ssim)
                ssim_diff = ssim - ssim_gt if ssim_gt is not None else None
                ssim_diff_list.append(ssim_diff)
                
                lpips = cur['average'].get('LPIPS', None)
                lpips_list.append(lpips)
                lpips_diff = (lpips - lpips_gt) if (lpips is not None and lpips_gt is not None) else None
                lpips_diff_list.append(lpips_diff)

        df = pd.DataFrame({
            'frame': frame_list,
            'psnr': psnr_list,
            'psnr_gt': psnr_gt_list,
            'psnr_diff': psnr_diff_list,
            'ssim': ssim_list,
            'ssim_gt': ssim_gt_list,
            'ssim_diff': ssim_diff_list,
            'lpips': lpips_list,
            'lpips_gt': lpips_gt_list,
            'lpips_diff': lpips_diff_list
        })

        # Save DataFrame as CSV in base_dir
        csv_path = os.path.join(base_dir_cur, f'{scene}_quality_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved quality summary table for {scene} to {csv_path}')

        # Compute average and save as CSV
        avg = df.mean(numeric_only=True)
        avg_path = os.path.join(base_dir_cur, f'{scene}_quality_average.csv')
        avg.to_frame(name='average').to_csv(avg_path)
        print(f'Saved quality average table for {scene} to {avg_path}')

        # Collect scene average for global average
        scene_avg_dict = avg.to_dict()
        scene_avg_dict['scene'] = scene
        scene_averages.append(scene_avg_dict)

    # Compute overall average across all scenes and save
    if scene_averages:
        avg_df = pd.DataFrame(scene_averages)
        avg_numeric = avg_df.drop(columns=['scene']).mean()
        avg_numeric['scene'] = 'overall_average'
        avg_df = avg_df._append(avg_numeric, ignore_index=True)
        overall_avg_path = os.path.join(base_dir, f'{dataset}_all_scenes_quality_average.csv')
        avg_df.to_csv(overall_avg_path, index=False)
        print(f'Saved overall quality average table to {overall_avg_path}')

if __name__ == "__main__":
    main()
