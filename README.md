# [ARXIV'25] D-FCGS
Official Pytorch Implementation of **D-FCGS: Feedforward Compression of Dynamic Gaussian Splatting for Free-Viewpoint Videos**.

[Wenkang Zhang](https://mr-zwkid.github.io/), 
[Yao Zhao](https://github.com/adminasmi), 
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

<!-- We tested our code on a server with Ubuntu 20.04.1, cuda 11.8, gcc 9.4.0. We use NVIDIA L40s GPU (48G).
 -->


## Run
FCGS can *directly* compress any existing 3DGS representations to bitstreams. The input should be a *.ply* file following the 3DGS format.

### To compress a *.ply* file to bitstreams, run:

```
python encode_single_scene.py --lmd A_lambda --ply_path_from PATH/TO/LOAD/point_cloud.ply --bit_path_to PATH/TO/SAVE/BITSTREAMS --determ 1
```
 - ```lmd```: the trade-off parameter for size and fidelity. Chosen in [```1e-4```, ```2e-4```, ```4e-4```, ```8e-4```, ```16e-4```].
 - ```ply_path_from```: A *.ply* file. Path to load the source *.ply* file.
 - ```bit_path_to```: A directory. Path to save the compressed bitstreams.
 - ```determ```: see [atomic statement](https://github.com/YihangChen-ee/FCGS/blob/main/docs/atomic_statement.md)

### To decompress a *.ply* file from bitstreams, run:

```
python decode_single_scene.py --lmd A_lambda --bit_path_from PATH/TO/LOAD/BITSTREAMS --ply_path_to PATH/TO/SAVE/point_cloud.ply
```
 - ```lmd```: the trade-off parameter for size and fidelity. Chosen in [```1e-4```, ```2e-4```, ```4e-4```, ```8e-4```, ```16e-4```].
 - ```bit_path_from```: A directory. Path to load the compressed bitstreams.
 - ```ply_path_to```: A *.ply* file. Path to save the decompressed *.ply* file.

### To decompress a *.ply* file from bitstreams and validate fidelity of the decompressed 3DGS, run:

```
python decode_single_scene_validate.py --lmd A_lambda --bit_path_from PATH/TO/LOAD/BITSTREAMS --ply_path_to PATH/TO/SAVE/point_cloud.ply --source_path PATH/TO/SOURCE/SCENES
```
 - ```source_path```: A directory. Path to load the source scene images for validation.

### Tips
FCGS is compatible with pruning-based techniques such as [Mini-Splatting](https://github.com/fatPeter/mini-splatting) and [Trimming the fat](https://github.com/salmanali96/trimming-the-fat). You can *directly* apply FCGS to the *.ply* file output by these two approaches to further boost the compression performance.

## CUDA accelerated arithmetic codec
We alongside publish a CUDA-based arithmetic codec implementation (based on [torchac](https://github.com/fab-jul/torchac)), you can find it in [arithmetic](https://github.com/YihangChen-ee/FCGS/blob/main/submodules/arithmetic) and its usage [here](https://github.com/YihangChen-ee/FCGS/blob/main/model/encodings_cuda.py).

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