# [ARXIV'25] D-FCGS
Official Pytorch Implementation of **D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos**.

[Wenkang Zhang](https://mr-zwkid.github.io/), 
[Yan Zhao](https://github.com/adminasmi), 
[Qiang Wang](https://scholar.google.com/citations?user=17E9fdUAAAAJ&hl=en), 
[Zhengxue Cheng](https://medialab.sjtu.edu.cn/author/zhengxue-cheng/)

<!-- [[`Arxiv`](https://arxiv.org/pdf/2410.08017)] [[`Project`](https://yihangchen-ee.github.io/project_fcgs/)] [[`Github`](https://github.com/YihangChen-ee/FCGS)] -->


## Overview
<p align="left">
<img src="assets/teaser.png" width=100%
class="center">
</p>

Left: Existing GS-based methods for FVV often couple scene reconstruction with compression and requireper scene optimization, resulting in reduced generalizability. In contrast,our D-FCGS decouples these stages with a single feedforward
 pass that compresses inter-frame motion in Gaussian frames,enabling efficient compression and storage for FFV. Right: Despite
 being optimization-free, D-FCGS achieves competitive rate-distortion performance compared to optimization-based methods.

## Method
<p align="left">
<img src="assets/method.png" width=100%
class="center">
</p>


## Installation

1. clone our code
   ```bash
   git clone https://github.com/Mr-Zwkid/FCGS-D.git --recursive  
   ```

2. create conda env and enter it
   ```bash
   conda create -n dfcgs python=3.10
   conda activate dfcgs
   ```

3. install pakages
   ```bash
   conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
   pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   pip install pytorch3d lpips tqdm plyfile 
   pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
   pip install submodules/simple-knn
   pip install submodules/diff-gaussian-rasterization
   pip install submodules/arithmetic
   ```
   
## Dataset Preprocess
1. Multiview Video Datasets
   - [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0)
   - [Meetroom](https://www.modelscope.cn/datasets/DAMOXR/dynamic_nerf_meeting_room_dataset/files)
   - [Google Immersive](https://github.com/augmentedperception/deepview_video_dataset?tab=readme-ov-file)
   - [WideRange4D](https://huggingface.co/datasets/Gen-Verse/WideRange4D)

2. Extract Frames
   Assume the path to the scene is `data_video/Immersive/01_Welder`. Extract the frames using
      ```python
      python extract_frames.py --base_dir data_video/Immersive/01_Welder [--scale_factor 2]
      ```

3. Colmap
   - Initial Frame
      ```python
      python convert_base.py -s data_video/Immersive/01_Welder/frame000000
      ```
   - Subsequent Frames
      ```python
      python copy_cams.py --source data_video/Immersive/01_Welder/frame000000 --scene  data_video/Immersive/01_Welder
      python convert_frames.py -s data_video/Immersive/01_Welder [--last_frame_id 299]
      ```

   

## Run


## Contact

- Wenkang Zhang: conquer.wkzhang@sjtu.edu.cn

## Citation

<!-- ```bibtex
@article{fcgs2024,
  title={Fast Feedforward 3D Gaussian Splatting Compression},
  author={Chen, Yihang and Wu, Qianyi and Li, Mengyao and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  journal={arXiv preprint arXiv:2410.08017},
  year={2024}
}
``` -->


## Acknowledgement

 - We thank authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
 - We thank authors from [3DGStream](https://github.com/SJoJoK/3DGStream) for extending 3DGS to a streamable version, thus providing a simple way of generating sequential Gaussian frames.
 - We thank authors from [FCGS](https://github.com/YihangChen-ee/FCGS) for their pioneering work on feedforward compression of static Gaussian Splatting.