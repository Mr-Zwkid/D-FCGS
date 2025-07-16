"""
Multi-camera video frame extraction and reorganization
"""
import os
import cv2
from tqdm import tqdm
import argparse
import shutil


def extract_and_reorganize_frames(base_dir, scale_factor, start_frame=None, end_frame=None, subdir = ''):
    """
    Extract frames from multi-camera videos, resize, and reorganize them into per-frame folders.
    If start_frame and end_frame are specified, use their indices for naming and folder creation, and do not recount the minimum number.
    """
    cam_mp4s = sorted([f for f in os.listdir(base_dir) if f.endswith('.mp4')])
    cam_names = [os.path.splitext(f)[0] for f in cam_mp4s]
    tmp_dirs = []
    with tqdm(total=len(cam_names), desc='Cameras', position=0, unit='camera') as cam_pbar:
        for cam, mp4 in zip(cam_names, cam_mp4s):
            cam_dir = os.path.join(base_dir, f'{cam}_frames')
            tmp_dirs.append(cam_dir)
            os.makedirs(cam_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(os.path.join(base_dir, mp4))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            s_frame = start_frame if start_frame is not None else 0
            e_frame = end_frame if end_frame is not None else total_frames - 1
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx < s_frame or frame_idx > e_frame:
                    continue
                h, w = frame.shape[:2]
                frame_small = cv2.resize(frame, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(cam_dir, f'frame{frame_idx:06d}.png'), frame_small)
            cap.release()
            cam_pbar.update(1)

    # Determine frame indices to use
    if start_frame is not None and end_frame is not None:
        frame_indices = range(start_frame, end_frame + 1)
        print(f'Extracting frames from {start_frame} to {end_frame} (total: {end_frame - start_frame + 1})')
    else:
        # Fallback: count the minimum number of frames among all cameras
        frame_lists = [sorted([int(f[5:11]) for f in os.listdir(cam_dir) if f.endswith('.png')]) for cam_dir in tmp_dirs]
        min_len = min(len(lst) for lst in frame_lists)
        frame_indices = [lst[:min_len] for lst in frame_lists][0]  # Use the first camera's indices
        print(f'Total Frame Number: {min_len}')

    # Create frame folders and reorganize images
    for i, frame_idx in enumerate(frame_indices):
        frame_dir = os.path.join(base_dir, f'frame{frame_idx:06d}', subdir)
        os.makedirs(frame_dir, exist_ok=True)
        for cam_idx, cam_dir in enumerate(tmp_dirs):
            src = os.path.join(cam_dir, f'frame{frame_idx:06d}.png')
            dst = os.path.join(frame_dir, f'cam{cam_idx+1:02d}.png')
            if os.path.exists(src):
                os.rename(src, dst)

    # Optional: Delete temporary frame folders
    for cam_dir in tmp_dirs:
        shutil.rmtree(cam_dir)

    print('Done!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract and reorganize frames from multi-camera videos.')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing multi-camera videos.')
    parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for resizing frames (e.g., 2 for half size).')
    parser.add_argument('--start_frame', type=int, default=None, help='Start frame index (inclusive).')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (inclusive).')
    parser.add_argument('--subdir', type=str, default='inputs', help='Subdirectory to place camera images in each frame folder.')
    args = parser.parse_args()

    base_dir = args.base_dir
    scale_factor = args.scale_factor
    start_frame = args.start_frame
    end_frame = args.end_frame
    subdir = args.subdir

    extract_and_reorganize_frames(base_dir, scale_factor, start_frame, end_frame, subdir)