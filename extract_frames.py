"""
Multi-camera video frame extraction and reorganization
"""
import os
import cv2
from tqdm import tqdm
import argparse
# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract and reorganize frames from multi-camera videos.')
parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing multi-camera videos.')
parser.add_argument('scale_factor', type=int, default=2,  help='Scale factor for resizing frames (e.g., 2 for half size).')
args = parser.parse_args()

base_dir = args.base_dir
scale_factor = args.scale_factor

# 1. Extract frames for each camera to a temporary folder and scale to half size
cam_mp4s = sorted([f for f in os.listdir(base_dir) if f.endswith('.mp4')])
cam_names = [os.path.splitext(f)[0] for f in cam_mp4s]
tmp_dirs = []
for cam, mp4 in zip(cam_names, cam_mp4s):
    print(f'Processing camera: {cam}')
    cam_dir = os.path.join(base_dir, f'{cam}_frames')
    tmp_dirs.append(cam_dir)
    os.makedirs(cam_dir, exist_ok=True)
    cap = cv2.VideoCapture(os.path.join(base_dir, mp4))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        frame_small = cv2.resize(frame, (w // scale_factor, h // scale_factor), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(cam_dir, f'frame{idx:06d}.png'), frame_small)
        idx += 1
    cap.release()

# 2. Count the minimum number of frames among all cameras, in case some cameras have fewer frames
frame_counts = []
for cam_dir in tmp_dirs:
    frames = [f for f in os.listdir(cam_dir) if f.endswith('.png')]
    frame_counts.append(len(frames))
min_frames = min(frame_counts)
print(f'Total Frame Number: {min_frames}')

# 3. Create frame folders and reorganize images
for i in tqdm(range(min_frames), desc='Reorganizing frames'):
    frame_dir = os.path.join(base_dir, f'frame{i:06d}')
    os.makedirs(frame_dir, exist_ok=True)
    for cam_idx, cam_dir in enumerate(tmp_dirs):
        src = os.path.join(cam_dir, f'frame{i:06d}.png')
        dst = os.path.join(frame_dir, f'cam{cam_idx+1:02d}.png')
        if os.path.exists(src):
            os.rename(src, dst)

# 4. Optional: Delete temporary frame folders
import shutil
for cam_dir in tmp_dirs:
    shutil.rmtree(cam_dir)

print('Done!')