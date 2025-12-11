import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import argparse

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize compression results")
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

        motion_size_list = []
        prior_size_list = []
        total_size_list = []
        psnr_list = []
        ssim_list = []

        size_gt_list = []
        psnr_gt_list = []
        ssim_gt_list = []
        frame_list = []

        for frame in range(frame_start, frame_end):
            if frame % gof_size == 0:
                continue
            frame_list.append(frame)

            path_size = f'{base_dir_cur}/frame{frame:06d}/size.json'
            with open(path_size, 'r') as f:
                size = json.load(f)
                motion_size = size['bits_motion']
                prior_size = size['bits_prior_motion']
                total_size = motion_size + prior_size
                motion_size_list.append(motion_size)
                prior_size_list.append(prior_size)
                total_size_list.append(total_size)

            path_psnr = f'{base_dir_cur}/frame{frame:06d}/rendering_info.json'
            with open(path_psnr, 'r') as f:
                cur = json.load(f)
                psnr = cur['average']['PSNR']
                psnr_list.append(psnr)

                ssim = cur['average']['SSIM']
                ssim_list.append(ssim)
                
            try:
                gt_path_psnr = f'{gt_dir}/frame{frame:06d}/gs/rendering_info.json'
                with open(gt_path_psnr, 'r') as f:
                    gt = json.load(f)
                    psnr_gt = gt['average']['PSNR']
                    psnr_gt_list.append(psnr_gt)
                    ssim_gt = gt['average']['SSIM']
                    ssim_gt_list.append(ssim_gt)
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

            gt_path_size = f'{gt_dir}/frame{frame:06d}/gs/NTC.pth'
            gt_size = os.path.getsize(gt_path_size) / 1024 / 1024  # Convert to MB
            size_gt_list.append(gt_size)
        df = pd.DataFrame({
            'frame': frame_list,
            'motion_size': motion_size_list,
            'prior_size': prior_size_list,
            'total_size': total_size_list,
            'size_gt': size_gt_list,
            'psnr': psnr_list,
            'psnr_gt': psnr_gt_list,
            'ssim': ssim_list,
            'ssim_gt': ssim_gt_list
        })

        # Save DataFrame as CSV in base_dir
        csv_path = os.path.join(base_dir_cur, f'{scene}_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f'Saved summary table for {scene} to {csv_path}')

        # Compute average and save as CSV
        avg = df.mean(numeric_only=True)
        avg_path = os.path.join(base_dir_cur, f'{scene}_average.csv')
        avg.to_frame(name='average').to_csv(avg_path)
        print(f'Saved average table for {scene} to {avg_path}')

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
        overall_avg_path = os.path.join(base_dir, f'{dataset}_all_scenes_average.csv')
        avg_df.to_csv(overall_avg_path, index=False)
        print(f'Saved overall average table to {overall_avg_path}')

if __name__ == "__main__":
    main()