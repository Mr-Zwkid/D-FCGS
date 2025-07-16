import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Derive initial 3DGS from multi-camera videos.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Base directory containing multi-camera videos.')
    parser.add_argument('--scene_list', nargs='*', default=None, help='List of scene names to process.')
    
    args = parser.parse_args()
    
    dataset_dir = args.dataset_dir
    scene_list = os.listdir(dataset_dir) if args.scene_list is None else args.scene_list
    for scene in scene_list:
        scene_dir = os.path.join(dataset_dir, scene)

        os.system(f'python 3DGStream/train.py  -m {scene_dir}/frame000000/gs -s {scene_dir}/frame000000\
                    --iterations 4000 --densify_until_iter 3000 --sh_degree 3 --images images' )
        # os.system(f'python 3DGStream/render.py  -m {scene_dir}/frame000000/gs -s {scene_dir}/frame000000 --images images  --iteration 4000' )
        # os.system(f'python 3DGStream/metrics.py  -m {scene_dir}/frame000000/gs' )

if __name__ == "__main__":
    main()