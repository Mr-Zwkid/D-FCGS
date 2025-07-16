import os
import argparse
import sys

orig_cwd = os.getcwd()
gs_dir = os.path.abspath('./3DGStream')
os.chdir(gs_dir)
sys.path.insert(0, gs_dir)

def main():
    parser = argparse.ArgumentParser(description='Derive initial 3DGS from multi-camera videos.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory containing multi-camera videos.')
    parser.add_argument('--scene_list', nargs='*', default=None, help='List of scene names to process.')
    parser.add_argument('--start_frame', type=int, default=1, help='Start frame index (inclusive).')
    parser.add_argument('--end_frame', type=int, default=299, help='End frame index (inclusive).')
    parser.add_argument('--first_load_iteration', type=int, default=4000, help='Iteration to load for the initial frame.')
    
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    scene_list = os.listdir(dataset_dir) if args.scene_list is None else args.scene_list
    s = args.start_frame
    e = args.end_frame
    first_load_iteration = args.first_load_iteration
    for scene in scene_list:
        scene_dir = os.path.join(dataset_dir, scene)

        cmd = f"python train_frames.py \
                        --read_config \
                        --config_path configs/cfg_args.json \
                        -o {scene_dir} \
                        -m {scene_dir}/frame000000/gs \
                        -v {scene_dir} \
                        --image images \
                        --first_load_iteration {first_load_iteration} \
                        --ntc_path ntc/{scene}_ntc_params_F_4.pth \
                        --frame_start {s} \
                        --frame_end {e + 1} \
                        --sh_degree 3 \
                        --eval \
                "
        os.system(cmd)
if __name__ == "__main__":
    main()
    os.chdir(orig_cwd)