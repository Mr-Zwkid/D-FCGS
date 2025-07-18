import argparse
import os
import shutil
from tqdm import tqdm
from utils.extract_frames import extract_and_reorganize_frames

def copy_sparse_to_frames(source, scene):
    sparse_dir = os.path.join(source, 'sparse')
    assert os.path.isdir(sparse_dir), f"Error: The directory '{sparse_dir}' does not exist."

    for item in os.listdir(scene):
        frame_dir = os.path.join(scene, item)
        if os.path.isdir(frame_dir) and item.startswith('frame') and not item.endswith('000000'):
            dest_sparse_dir = os.path.join(frame_dir, 'sparse')
            if os.path.exists(dest_sparse_dir):
                shutil.rmtree(dest_sparse_dir)
            shutil.copytree(sparse_dir, dest_sparse_dir)
            print(f"Copied to {dest_sparse_dir}")

def copy_distorted_to_scene(source, scene):
    distorted_dir = os.path.join(source, 'distorted')
    assert os.path.isdir(distorted_dir), f"Error: The directory '{distorted_dir}' does not exist."

    dest_distorted_dir = os.path.join(scene, 'distorted')
    if os.path.exists(dest_distorted_dir):
        shutil.rmtree(dest_distorted_dir)
    shutil.copytree(distorted_dir, dest_distorted_dir)
    print(f"Copied to {dest_distorted_dir}")

def get_colmap_single(scene_path, offset, subdir='inputs'):
    if offset == 0:
        frame_path = os.path.join(scene_path, "frame000000")
        os.system(f'python utils/convert.py -s {frame_path} --image_dir {subdir}')
        
        # Copy sparse and distorted directories to each frame
        copy_sparse_to_frames(frame_path, scene_path)
        copy_distorted_to_scene(frame_path, scene_path)
    else:
        frame_path = os.path.join(scene_path, f"frame{offset:06d}")
        input_image_folder = os.path.join(frame_path, subdir)
        distorted = os.path.join(scene_path, "distorted/sparse/0")
                    
        img_undist_cmd = (f"colmap image_undistorter \
                        --image_path {input_image_folder} \
                        --input_path {distorted} \
                        --output_path {frame_path} \
                        --output_type COLMAP")
        os.system(img_undist_cmd)
        os.system(f"rm -r {input_image_folder}")

        files = os.listdir(frame_path + "/sparse")
        os.makedirs(frame_path + "/sparse/0", exist_ok=True)
        for file in files:
            if file == '0':
                continue
            source_file = os.path.join(frame_path, "sparse", file)
            destination_file = os.path.join(frame_path, "sparse", "0", file)
            shutil.move(source_file, destination_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and reorganize frames from multi-camera videos.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory containing multi-camera videos.')
    parser.add_argument('--scene_list', nargs='*', default=None, help='List of scene names to process.')

    parser.add_argument('--scale_factor', type=int, default=2, help='Scale factor for resizing frames (e.g., 2 for half size).')
    parser.add_argument('--start_frame', type=int, default=None, help='Start frame index (inclusive).')
    parser.add_argument('--end_frame', type=int, default=None, help='End frame index (inclusive).')
    parser.add_argument('--subdir', type=str, default='inputs', help='Subdirectory to place camera images in each frame folder.')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    scale_factor = args.scale_factor
    start_frame = args.start_frame
    end_frame = args.end_frame
    subdir = args.subdir

    for scene in args.scene_list:
        base_dir = os.path.join(dataset_dir, scene)
        
        # 1. Extract frames and reorganize them
        extract_and_reorganize_frames(base_dir, scale_factor, start_frame, end_frame, subdir)

        # 2. Colmap processing
        for i in tqdm(range(start_frame, end_frame + 1), desc='Processing frames'):
            get_colmap_single(base_dir, i, subdir=subdir)