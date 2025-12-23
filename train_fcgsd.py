import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import os
import json
import torchvision
import time
from scene import GaussianModel, Scene
from model.DFCGS_model import FCGS_D
from fcgs.FCGS_model import FCGS
from utils.zstd_xyz_codec import zstd_decompress_xyz_from_file, zstd_roundtrip_xyz, zstd_roundtrip_xyz_quantized

from gaussian_renderer import render
import lpips
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

from utils.logger import Logger
import glob
from plyfile import PlyData, PlyElement 
import numpy as np
from typing import NamedTuple

lpips_fn = None


def _summarize_motion_xyz(xyz: torch.Tensor, tag: str = "motion_xyz") -> None:
    if xyz is None:
        print(f"[{tag}] None")
        return
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.as_tensor(xyz)
    if xyz.numel() == 0:
        print(f"[{tag}] empty")
        return

    x = xyz.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().reshape(-1)
    absx = np.abs(x)

    mn, mx = float(x.min()), float(x.max())
    mean, std = float(x.mean()), float(x.std())

    ps = [50, 75, 90, 95, 99, 99.9]
    pvals = np.percentile(absx, ps)

    thr = [1e-6, 1e-5, 1e-4, 1e-3]
    ratios = [float((absx <= t).mean()) for t in thr]
    zero_ratio = float((absx == 0).mean())

    print(f"[{tag}] shape={tuple(xyz.shape)} min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g}")
    print(
        f"[{tag}] |x| percentiles: "
        + ", ".join([f"p{p}={v:.6g}" for p, v in zip(ps, pvals)])
    )
    print(
        f"[{tag}] mass near 0: "
        + ", ".join([f"|x|<={t:g}:{r*100:.2f}%" for t, r in zip(thr, ratios)])
        + f", |x|==0:{zero_ratio*100:.2f}%"
    )


def _update_size_json_with_motion_xyz(frame_path: str, zst_path: str, logger: Logger | None = None) -> tuple[int, int]:
    """Update frame{xxxxxx}/size.json with motion xyz zstd bitstream size.

    Returns (bytes, bits). If size.json is missing, it is silently skipped.
    """
    if not zst_path or not os.path.exists(zst_path):
        return 0, 0

    motion_xyz_bytes = int(os.path.getsize(zst_path))
    motion_xyz_mb = float(motion_xyz_bytes / (1024 * 1024))

    size_json_path = os.path.join(frame_path, 'size.json')
    if os.path.exists(size_json_path):
        try:
            with open(size_json_path, 'r') as f:
                size = json.load(f)
        except Exception:
            size = {}

        # NOTE: bits_motion/bits_prior_motion/bits_total are historically stored in MB (not bits).
        size['bytes_motion_xyz_zst'] = motion_xyz_bytes
        size['motion_xyz_size_mb'] = motion_xyz_mb

        # Default policy: bits_total (MB) should include motion_xyz_size_mb.
        # Preserve the original model-reported bits_total as bits_total_model_mb.
        if 'bits_total_model_mb' not in size and 'bits_total' in size:
            try:
                size['bits_total_model_mb'] = float(size.get('bits_total', 0.0))
            except Exception:
                size['bits_total_model_mb'] = 0.0

        try:
            base_total_mb = float(size.get('bits_total_model_mb', size.get('bits_total', 0.0)))
        except Exception:
            base_total_mb = 0.0
        size['bits_total'] = base_total_mb + motion_xyz_mb

        # Remove old/incorrect fields if present.
        size.pop('bits_motion_xyz_zst', None)
        size.pop('bits_total_with_motion_xyz', None)

        with open(size_json_path, 'w') as f:
            json.dump(size, f, indent=True)

    if logger is not None:
        try:
            logger.log_info(f"motion_xyz zstd size: {motion_xyz_bytes} bytes ({motion_xyz_mb:.6f} MB)")
        except Exception:
            pass
    return motion_xyz_bytes, motion_xyz_mb


def _zstd_roundtrip_xyz(
    origin_motion_xyz: torch.Tensor,
    zst_path: str,
    zstd_bin: str,
    zstd_level: int,
    zstd_long: int | None,
    zstd_threads: int,
    motion_xyz_quant: str,
    motion_xyz_quant_step: float,
    motion_xyz_deadzone: float | None,
    motion_xyz_quant_bias: str,
    logger: Logger,
) -> torch.Tensor:
    """Compress + decompress xyz via zstd and return the reconstructed xyz tensor.

    Preserves point order. Can be lossless (quant=none) or lossy (quant != none).
    """
    if origin_motion_xyz is None:
        return None
    if not isinstance(origin_motion_xyz, torch.Tensor):
        origin_motion_xyz = torch.as_tensor(origin_motion_xyz)

    # Ensure shape is (N, 3)
    if origin_motion_xyz.ndim != 2 or origin_motion_xyz.shape[-1] != 3:
        raise ValueError(f"origin_motion_xyz must have shape (N,3), got {tuple(origin_motion_xyz.shape)}")

    os.makedirs(os.path.dirname(zst_path), exist_ok=True)
    if motion_xyz_quant is None or motion_xyz_quant == 'none':
        origin_motion_xyz_dec = zstd_roundtrip_xyz(
            origin_motion_xyz,
            out_zst_path=zst_path,
            device=origin_motion_xyz.device,
            zstd_bin=zstd_bin,
            level=zstd_level,
            long=zstd_long,
            threads=zstd_threads,
        )
    else:
        origin_motion_xyz_dec = zstd_roundtrip_xyz_quantized(
            origin_motion_xyz,
            out_zst_path=zst_path,
            device=origin_motion_xyz.device,
            quant_mode=motion_xyz_quant,
            step=float(motion_xyz_quant_step),
            deadzone=motion_xyz_deadzone,
            bias_mode=motion_xyz_quant_bias,
            zstd_bin=zstd_bin,
            level=zstd_level,
            long=zstd_long,
            threads=zstd_threads,
        )
    if logger is not None:
        try:
            logger.log_info(f"zstd xyz bitstream saved: {zst_path} ({os.path.getsize(zst_path)} bytes)")
        except Exception:
            logger.log_info(f"zstd xyz bitstream saved: {zst_path}")
    return origin_motion_xyz_dec

def get_lpips_fn(device):
    """Return a cached LPIPS model on the requested device."""
    global lpips_fn
    if lpips_fn is None or next(lpips_fn.parameters()).device != device:
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
    return lpips_fn

def plot_loss_curve(render_loss, size_loss, total_loss, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(render_loss, label='render_loss')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Render Loss Curve')
    axes[0].legend()
    
    axes[1].plot(size_loss, label='size_loss')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Size Loss Curve')
    axes[1].legend()
    
    axes[2].plot(total_loss, label='total_loss')
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Total Loss Curve')
    axes[2].legend()
    
    plt.tight_layout()

    plt.savefig(save_path)
    plt.show()
    plt.clf()

def read_gaussian_file( file_path, sh_degree = 3):
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(file_path)
    return gaussians

def path_match(pattern):
    matches = glob.glob(pattern)
    if matches:
        return sorted(matches)[-1]  # pick the last one (highest iteration)
    else:
        raise FileNotFoundError(f"No matching files found for pattern: {pattern}")

def train_frame_setup(args, frame_cur, frame_next=None):
    
    if args.dynamicGS_type == '3dgstream_explicit' or args.dynamicGS_type == '3dgstream_implicit' or  args.dynamicGS_type == 'control_point':
        for dataset in args.dataset_list:
            base_dir = f'{args.dataset_path}/{dataset}/{args.scene_name}'
            if os.path.exists(base_dir):
                break
        else:
            raise FileNotFoundError(f"Dataset {args.dataset_list} not found in {args.dataset_path}/{args.scene_name}")
        
        if frame_cur >= 0:
            if frame_cur % args.gof_size ==0 and args.use_I_compress: # I frame
                lmd=args.lmd
                if args.use_added and frame_cur !=0:
                    ply_path = f'{base_dir}/frame{frame_cur:06d}/FCGS_point_cloud_added_{lmd}.ply'
                else:
                    ply_path = f'{base_dir}/frame{frame_cur:06d}/FCGS_point_cloud_{lmd}.ply'
                    if not os.path.exists(ply_path):
                        raise FileNotFoundError(f"FCGS ply file not found at {ply_path} for frame {frame_cur}")
                args.init_3dgs = ply_path
            else:
                init_pattern = f'{base_dir}/frame{frame_cur:06d}/gs/point_cloud/iteration_*/point_cloud.ply'
                args.init_3dgs = path_match(init_pattern)

        frame_next = frame_cur + 1 if frame_next is None else frame_next
        next_pattern = f'{base_dir}/frame{frame_next:06d}/gs/point_cloud/iteration_*/point_cloud.ply'
        args.next_3dgs = path_match(next_pattern)

        if frame_next != 0:
            args.added_gs = f'{base_dir}/frame{frame_next:06d}/gs/point_cloud/iteration_250/added/point_cloud.ply'
        
        args.source_path = f'{base_dir}/frame{frame_next:06d}' # the source path for the next frame images, used for evaluation
        args.frame_path = os.path.join(args.model_path, args.scene_name, f'frame{frame_next:06d}')
    
        args.FCGS_path = args.source_path
    else:
        raise ValueError(f"Unknown dynamicGS_type: {args.dynamicGS_type}")

# Final Version for Training
def train_frame_gof(args):
    logger = args.logger
    logger.log_info("Starting training within GoF...")

    model = FCGS_D(args).cuda()

    # Checkpoint Loading
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        logger.log_info(f"Loaded checkpoint from {args.checkpoint_path}")
    
    model.train()

    render_loss = []
    size_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")
    buffer_gaussian = None

    PSNR_train_list = []
    PSNR_test_list = []


    for i in pbar:
        # set frame/scene to train
        with torch.no_grad():
            if args.use_scenes and i % args.gof_size == 0: # change scene and select new start frame
                args.scene_name = args.scene_list[torch.randint(0, len(args.scene_list), (1,)).item()] # randomly select a scene
                frame_cur = torch.randint(args.frame_start, args.frame_end, (1,)).item() // args.gof_size * args.gof_size # randomly select a start frame, which is a multiple of args.gof_size
                frame_next = frame_cur + 1
                train_frame_setup(args, frame_cur, frame_next)
                model.refresh_settings(args)
            else: # step forward and update buffer gaussians
                frame_cur = min(frame_cur + 1, args.frame_end - 2)
                frame_next = frame_cur + 1
                train_frame_setup(args, frame_cur, frame_next)
                model.buffer_loading(args, buffer_gaussian)
        

        loss_render, loss_size, loss_motion, PSNR_train, PSNR_test = model()
        loss_size = loss_size if loss_size is not None else torch.tensor(0.0).to('cuda')

        lambda_size = args.lambda_size  if i > args.iterations // 3 * 2 else 0.0
        # lambda_distortion = 1 / lambda_size
        loss = loss_render + lambda_size * loss_size + loss_motion * 10

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        torch.cuda.empty_cache()

        render_loss.append(loss_render.item())
        size_loss.append(loss_size.item())
        total_loss.append(loss.item())

        # logging
        logger.log_iteration(i, {
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_motion': loss_motion.item() if hasattr(loss_motion, 'item') else loss_motion,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss,
            'PSNR_train': PSNR_train,
            'PSNR_test': PSNR_test
        }, additional_info={'frame_cur': frame_cur, 'frame_next': frame_next})

        PSNR_train_list.append(PSNR_train)
        PSNR_test_list.append(PSNR_test)

        logger.log_info(f"Iteration {i}: loss_render={loss_render.item():.4f}, loss_size={loss_size.item():.4f}, loss_motion={loss_motion.item():.4f}, total_loss={loss.item():.4f}, PSNR_train={PSNR_train:.4f}, PSNR_test={PSNR_test:.4f}")

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_motion': loss_motion.item() if hasattr(loss_motion, 'item') else loss_motion,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        # save checkpoint
        if args.checkpoint_iteration is not None and i % args.checkpoint_iteration == 0:
            model_path = os.path.join(args.model_path, args.scene_name, f'checkpoint_{i}.pth')
            logger.log_info(f"Saving checkpoint to {model_path}")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)

        # update buffer frame
        with torch.no_grad():
            buffer_gaussian = model.buffer_capture()

    model_path = os.path.join(args.model_path, args.scene_name, 'model.pth')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.log_info(f"Saved final model to {model_path}")

    loss_curve_path = os.path.join(args.model_path, args.scene_name, 'loss_curve.png')
    plot_loss_curve(render_loss, size_loss, total_loss, loss_curve_path)
    logger.log_info(f"Saved loss curves to {loss_curve_path}")

    # save PSNR csv and compute average PSNR
    psnr_csv_path = os.path.join(args.model_path, args.scene_name, 'psnr_curve.csv')
    with open(psnr_csv_path, 'w') as f:
        f.write('Iteration,PSNR_train,PSNR_test\n')
        for idx in range(len(PSNR_train_list)):
            f.write(f"{idx},{PSNR_train_list[idx]},{PSNR_test_list[idx]}\n")
    logger.log_info(f"Saved PSNR curves to {psnr_csv_path}")

    logger.log_training_complete()

def debug_control_point(args, frame_start, frame_end):
    logger = args.logger
    logger.create_list_for_logging('compress_time')
    logger.create_list_for_logging('decompress_time')
    logger.create_list_for_logging('psnr')
    logger.create_list_for_logging('ssim')
    logger.create_list_for_logging('lpips')
    logger.create_list_for_logging('total_size')
    for frame in range(frame_start, frame_end + 1):
        train_frame_setup(args, frame-1, frame)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene
        print('pre_frame:', frame-1)
        print('cur_frame:', frame)
        print(args.init_3dgs)
        print(args.next_3dgs)
        print(args.frame_path)
        os.makedirs(args.frame_path, exist_ok=True)
        if frame % args.gof_size == 0:
            print(f"Compressing frame {frame} as I frame (Currently Skipping)...")
        elif frame % args.gof_size == 1:
            buffer_gaussian, ctrl_point_idx = myFCGS_D.debug_control_point(use_buffer = False, buffer_gaussian = None, cur_gaussians_path=args.init_3dgs, nxt_gaussians_path=args.next_3dgs)
            validate(buffer_gaussian, scene, args,  args.frame_path, save_img=True, logger=logger)
        else:
            buffer_gaussian, _ = myFCGS_D.debug_control_point(use_buffer = True, buffer_gaussian = buffer_gaussian, cur_gaussians_path=args.init_3dgs, nxt_gaussians_path=args.next_3dgs, ctrl_point_idx=ctrl_point_idx)
            validate(buffer_gaussian, scene, args,  args.frame_path, save_img=True, logger=logger)
        
    return buffer_gaussian

def merge_2ply(ply_path1: str, ply_path2: str, out_path: str):
    """
    合并两个 3D-GS 生成的 ply 文件（stage1 + stage2_added 等）
    输出可直接被 GaussianModel.load_ply 读取
    """
    # gauss_tmp = GaussianModel(3)
    # gauss_tmp.load_ply(ply_path2)
    # 1. 读取两个文件
    ply1 = PlyData.read(ply_path1)
    ply2 = PlyData.read(ply_path2)

    v1 = ply1['vertex'].data
    v2 = ply2['vertex'].data

    # 2. 保证属性完全一致
    # assert v1.dtype == v2.dtype, "两个PLY属性不一致，无法合并"

    # 3. 拼接结构化数组
    # print(v1.dtype)
    # print(v2.dtype)
    merged = np.empty(len(v1) + len(v2), dtype=v2.dtype)
    merged[:len(v1)] = v1
    merged[len(v1):]  = v2

    # 4. 写出
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    el = PlyElement.describe(merged, 'vertex')
    PlyData([el]).write(out_path)


def FCGS_compress(ply_path_from, bit_path_to, lmd,determ=1,nr=3):#
    # assert False, 'check scene dataloader first!'

    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
        gaussians.load_ply(path=ply_path_from)
    g_xyz = gaussians._xyz.detach()
    N_gaussian = g_xyz.shape[0]

    per_step_size = 100_0000
    if N_gaussian > 100_0000 and N_gaussian < 110_0000:
        per_step_size = 110_0000

    _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
    _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
    _opacity = gaussians._opacity.detach()  # [N, 1]
    _scaling = gaussians._scaling.detach()  # [N, 3]
    _rotation = gaussians._rotation.detach()  # [N, 4]
    g_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation], dim=-1)  # [N, 56]
    # 1. 构造「通道级」最大值（忽略 inf）
    finite_max = torch.where(torch.isinf(g_fea), torch.full_like(g_fea, -torch.inf), g_fea).max(0, keepdim=True)[0]


    # # --- 先找 inf 的位置 ---
    # inf_mask = torch.isinf(g_fea)          # [N, 56]  bool
    # inf_count = inf_mask.sum().item()      # 全局 inf 的总数
    # if inf_count > 0:
    #     # 哪些通道（属性维度）出现了 inf
    #     inf_channels = torch.where(inf_mask.any(0))[0]  # 长度 [<=56]
    #     # 对应回原始属性，方便打印名字
    #     attr_names = (
    #         ['features_dc'] * _features_dc.shape[1] +
    #         ['features_rest'] * _features_rest.shape[1] +
    #         ['opacity'] * _opacity.shape[1] +
    #         ['scaling'] * _scaling.shape[1] +
    #         ['rotation'] * _rotation.shape[1]
    #     )
    #     print(f'[inf report] 共 {inf_count} 个 inf，分布在如下通道：')
    #     for ch in inf_channels.cpu().numpy():
    #         print(f'  通道 {ch:02d} ({attr_names[ch]})')
    # else:
    #     print('[inf report] 没有检测到 inf。')
    # 2. 把 inf 替换成对应通道的最大值
    g_fea = torch.where(torch.isinf(g_fea), finite_max, g_fea)
    step_num = int(np.ceil(N_gaussian / per_step_size))
    lmd = lmd
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'/mnt/data3/ctx/FCGS/checkpoints/checkpoint_{lmd}.pkl'), strict=True)
    for name, param in CM.named_parameters():
        if torch.isnan(param).any():
            print(f'[NaN] {name}: {param.shape}  min={param.min().item():.4f} max={param.max().item():.4f}')
            exit(1)
    ttl_size = 0
    CM.eval()
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(bit_path_to, f"{lmd}/{s}")
            os.makedirs(bit_save_path, exist_ok=True)
            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size]
            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size]
            # print("g_xyz_in",g_xyz_in)
            # print("g_fea_in",g_fea_in)

            ttl_size += CM.compress(g_xyz_in, g_fea_in, root_path=bit_save_path, chunk_size_list=chunk_size_list, determ_codec=determ)[3]
    
    torch.cuda.synchronize(); t2 = time.time()
    print('time:', t2-t1)

    print(f"{ply_path_from} compressed! Save bitstreams to {bit_path_to}.")
    orig_size = os.path.getsize(ply_path_from)/1024/1024
    print(f"Original size: {orig_size:.4f} MB. Compressed size: {ttl_size:.4f} MB. Compression ratio: {orig_size/ttl_size:.4f} X")


class D1(NamedTuple):
    data_device: str
    eval: bool
    images: str
    lod: int
    model_path: str
    resolution: int
    sh_degree: int
    source_path: str
    white_background: bool
    use_first_as_test: bool


def FCGS_decompress_validate(args, ply_path_to, bit_path_from, source_path,lmd,determ=1,nr=3,json_path=''):
    print("source_path",source_path)
    logger = args.logger
    logger.log_info(f"Decompressing FCGS bin  from {bit_path_from}")
    
    
    dataset = D1(
        data_device='cuda',
        eval=True,
        images='images',
        lod=0,
        model_path="",
        resolution=-1,
        sh_degree=3,
        source_path=source_path,
        white_background=False,
        use_first_as_test=True
    )
    # pipeline = D2(
    #     convert_SHs_python=False,
    #     compute_cov3D_python=False,
    #     debug=False
    # )

    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
        scene = Scene(dataset, shuffle=False)
        views = scene.getTestCameras()

    step_num = len(os.listdir(os.path.join(bit_path_from, str(lmd))))
    lmd = lmd
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'/mnt/data3/ctx/FCGS/checkpoints/checkpoint_{lmd}.pkl'), strict=True)

    g_xyz_list = []
    g_fea_list = []
    CM.eval()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(bit_path_from, f"{lmd}/{s}")
            g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list)
            g_xyz_list.append(g_xyz_out)
            g_fea_list.append(g_fea_out)
            
    g_xyz = torch.cat(g_xyz_list, dim=0)
    g_fea = torch.cat(g_fea_list, dim=0)

    f_dc, f_rst, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3, 45, 1, 3, 4], dim=-1)
    gaussians._xyz = nn.Parameter(g_xyz)
    gaussians._features_dc = nn.Parameter(f_dc.view(-1, 1, 3))
    gaussians._features_rest = nn.Parameter(f_rst.view(-1, 15, 3))
    gaussians._opacity = nn.Parameter(op.view(-1, 1))
    gaussians._scaling = nn.Parameter(sc.view(-1, 3))
    gaussians._rotation = nn.Parameter(ro.view(-1, 4))

    gaussians.save_ply(ply_path_to)
    print(f"Decompressed ply file saved to {ply_path_to}!")
    
    with torch.no_grad():
        validate(gaussians, scene, args, args.FCGS_path, save_img=True, logger=args.logger,json_path=json_path)

def conduct_compress_decompress_as_gof(args, frame_start, frame_end, json_path='rendering_info.json'):
    logger = args.logger
    logger.create_list_for_logging('compress_time')
    logger.create_list_for_logging('decompress_time')
    logger.create_list_for_logging('psnr')
    logger.create_list_for_logging('ssim')
    logger.create_list_for_logging('lpips')
    logger.create_list_for_logging('total_size')

    logger.log_info("Starting compress and decompress as GoF...")
    for frame in range(frame_start, frame_end + 1):
        logger.log_info(f"Processing frame {frame}...")
        train_frame_setup(args, frame-1, frame)
        os.makedirs(args.frame_path, exist_ok=True)

        model_path = os.path.join(args.model_path, 'model.pth') if args.checkpoint_path is None else args.checkpoint_path
        motion_save_path = os.path.join(args.frame_path, 'motion.b')
        motion_prior_save_path = os.path.join(args.frame_path, 'motion_prior.b')
        mask_save_path = os.path.join(args.frame_path, 'mask.b')
        ply_save_path = os.path.join(args.frame_path, 'output.ply')

        if frame % args.gof_size == 0:
            if args.use_I_compress:
                lmd = args.lmd
                if args.use_added and frame != 0:
                    added_all_ply_path = os.path.join(args.FCGS_path, 'added_all.ply')
                    print("args.added_gs",args.added_gs)
                    merge_2ply(args.next_3dgs, args.added_gs, added_all_ply_path)
                    fcgs_input_ply = added_all_ply_path
                    fcgs_ply_save_path = os.path.join(args.FCGS_path, f'FCGS_point_cloud_added_{lmd}.ply')#FCGS_path = f'{base_dir}/frame{frame_next:06d}'
                    FCGS_json_path = f'FCGS_validate_added_{lmd}.json'
                    bit_save_path = os.path.join(args.FCGS_path, f'FCGS_bitstreams_added_{lmd}')
                else:
                    fcgs_input_ply = args.next_3dgs
                    fcgs_ply_save_path = os.path.join(args.FCGS_path, f'FCGS_point_cloud_{lmd}.ply')
                    FCGS_json_path=f'FCGS_validate_{lmd}.json'
                    bit_save_path = os.path.join(args.FCGS_path, f'FCGS_bitstreams_{lmd}')

                logger.log_info(f"Compressing frame {frame} as I frame...")

                print(f"Starting FCGS compressing to {bit_save_path}")
                FCGS_compress(fcgs_input_ply, bit_save_path, lmd=lmd, determ=1, nr=3)

                print(f"Starting FCGS decompressing and validate from {fcgs_ply_save_path}")
                FCGS_decompress_validate(args, fcgs_ply_save_path, bit_save_path, source_path=args.source_path, lmd=lmd, determ=1, nr=3,json_path=FCGS_json_path)
                args.init_3dgs = fcgs_ply_save_path
        elif frame % args.gof_size == 1:
            orgin_motion_xyz = conduct_compress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, args.next_3dgs)
            dec_motion_xyz = orgin_motion_xyz
            if orgin_motion_xyz is not None:
                if getattr(args, 'motion_xyz_stats', False):
                    _summarize_motion_xyz(orgin_motion_xyz, tag=f"origin_motion_xyz@frame{frame:06d}")
                zst_path = os.path.join(args.frame_path, 'origin_motion_xyz.zst')
                dec_motion_xyz = _zstd_roundtrip_xyz(
                    orgin_motion_xyz,
                    zst_path,
                    args.zstd_bin,
                    args.zstd_level,
                    args.zstd_long,
                    args.zstd_threads,
                    args.motion_xyz_quant,
                    args.motion_xyz_quant_step,
                    args.motion_xyz_deadzone,
                    args.motion_xyz_quant_bias,
                    logger,
                )
                _, motion_xyz_mb = _update_size_json_with_motion_xyz(args.frame_path, zst_path, logger=logger)
                # Note: conduct_compress() already appended res['bits_total'] into total_size.
                # We mutate the latest entry to include motion_xyz bits.
                try:
                    logger.general_info['total_size'][-1] = float(logger.general_info['total_size'][-1]) + float(motion_xyz_mb)
                except Exception:
                    pass
                print('mse of motion_xyz:', torch.mean((orgin_motion_xyz - dec_motion_xyz) ** 2).item())
            buffer_gaussian = conduct_decompress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, ply_save_path, origin_motion_xyz=dec_motion_xyz, json_path=json_path)
        else:
            orgin_motion_xyz = conduct_compress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, args.next_3dgs, use_buffer=True, buffer_gaussian=buffer_gaussian)
            dec_motion_xyz = orgin_motion_xyz
            if orgin_motion_xyz is not None:
                if getattr(args, 'motion_xyz_stats', False):
                    _summarize_motion_xyz(orgin_motion_xyz, tag=f"origin_motion_xyz@frame{frame:06d}")
                zst_path = os.path.join(args.frame_path, 'origin_motion_xyz.zst')
                dec_motion_xyz = _zstd_roundtrip_xyz(
                    orgin_motion_xyz,
                    zst_path,
                    args.zstd_bin,
                    args.zstd_level,
                    args.zstd_long,
                    args.zstd_threads,
                    args.motion_xyz_quant,
                    args.motion_xyz_quant_step,
                    args.motion_xyz_deadzone,
                    args.motion_xyz_quant_bias,
                    logger,
                )
                _, motion_xyz_mb = _update_size_json_with_motion_xyz(args.frame_path, zst_path, logger=logger)
                try:
                    logger.general_info['total_size'][-1] = float(logger.general_info['total_size'][-1]) + float(motion_xyz_mb)
                except Exception:
                    pass
                print('mse of motion_xyz:', torch.mean((orgin_motion_xyz - dec_motion_xyz) ** 2).item())
            buffer_gaussian = conduct_decompress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, ply_save_path, use_buffer=True, buffer_gaussian=buffer_gaussian, origin_motion_xyz=dec_motion_xyz, json_path=json_path)
    logger.save_general_info()

def conduct_compress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', nxt_gaussians_path=None, use_buffer=False, buffer_gaussian=None):
    logger = args.logger
    logger.log_info(f"Model from {model_path}")
    logger.log_info(f"Compressing {nxt_gaussians_path} with reference of {init_3dgs_path}")
    
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()

        res, duration = myFCGS_D.compress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path, mask_path, dynamicGS_type=args.dynamicGS_type, nxt_gaussians_path=nxt_gaussians_path, use_buffer=use_buffer, buffer_gaussian=buffer_gaussian, add_position_noise=args.add_position_noise)
        logger.add_to_list('compress_time', duration)
        logger.log_info(f"Compression time: {duration:.2f} seconds")
        orgin_motion_xyz = res.get('origin_motion_xyz', None)
        res.pop('origin_motion_xyz', None)  # remove from res before saving
        with open(os.path.join(args.frame_path, 'size.json'), 'w') as f:
            json.dump(res, f, indent=True)
        logger.add_to_list('total_size', res['bits_total'])
        
        logger.log_info(f'Compression Result: {res}') 

        
    return orgin_motion_xyz

def conduct_decompress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', save_ply_path = './output.ply', use_buffer=False, buffer_gaussian=None, cur_gaussians=None, origin_motion_xyz=None, json_path='rendering_info_decompress'):
    logger = args.logger
    logger.log_info(f"Decompressing next Gaussian Frame from {init_3dgs_path}")
    
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene

        rec_gaussians, duration = myFCGS_D.decompress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path, mask_path, dynamicGS_type=args.dynamicGS_type, use_buffer=use_buffer, buffer_gaussian=buffer_gaussian)
        logger.add_to_list('decompress_time', duration)
        logger.log_info(f"Decompression time: {duration:.2f} seconds")

        rec_gaussians._xyz += origin_motion_xyz.to(rec_gaussians._xyz.device) if origin_motion_xyz is not None else None
        # rec_gaussians.save_ply(save_ply_path)
        validate(rec_gaussians, scene, args,  args.frame_path, save_img=True, logger=logger, json_path=json_path)

    return rec_gaussians

def conduct_decompress_video(args, frame_start, frame_end, json_path = 'rendering_info_decompress'):
    print("-------------conduct_decomress_video----------------")
    logger = args.logger
    # logger.create_list_for_logging('compress_time')
    logger.create_list_for_logging('decompress_time')
    logger.create_list_for_logging('psnr')
    logger.create_list_for_logging('ssim')
    logger.create_list_for_logging('lpips')
    # logger.create_list_for_logging('total_size')
    
    logger.log_info("Starting  decompress as whole video...")
    for frame in range(frame_start, frame_end + 1):
        logger.log_info(f"Processing frame {frame}...")
        train_frame_setup(args, frame-1, frame)
        os.makedirs(args.frame_path, exist_ok=True)

        model_path = os.path.join(args.model_path, 'model.pth') if args.checkpoint_path is None else args.checkpoint_path
        motion_save_path = os.path.join(args.frame_path, 'motion.b')
        motion_prior_save_path = os.path.join(args.frame_path, 'motion_prior.b')
        mask_save_path = os.path.join(args.frame_path, 'mask.b')
        ply_save_path = os.path.join(args.frame_path, 'output_decompress.ply')


        if frame % args.gof_size == 0:
            print(f"pass I frame {frame}")
        elif frame % args.gof_size == 1:
            print(f"decompress frame {frame}", f"reference {args.init_3dgs}",motion_prior_save_path,motion_save_path)
            zst_path = os.path.join(args.frame_path, 'origin_motion_xyz.zst')
            origin_motion_xyz = None
            if os.path.exists(zst_path):
                origin_motion_xyz = zstd_decompress_xyz_from_file(
                    zst_path,
                    device=torch.device(f"cuda:{args.gpu}"),
                    zstd_bin=args.zstd_bin,
                )
            buffer_gaussian = conduct_decompress(
                args,
                model_path,
                args.init_3dgs,
                motion_save_path,
                motion_prior_save_path,
                mask_save_path,
                ply_save_path,
                origin_motion_xyz=origin_motion_xyz,
                json_path=json_path,
            )
        else:
            print(f"decompress frame {frame}", f"reference {args.init_3dgs}",motion_prior_save_path,motion_save_path)
            zst_path = os.path.join(args.frame_path, 'origin_motion_xyz.zst')
            origin_motion_xyz = None
            if os.path.exists(zst_path):
                origin_motion_xyz = zstd_decompress_xyz_from_file(
                    zst_path,
                    device=torch.device(f"cuda:{args.gpu}"),
                    zstd_bin=args.zstd_bin,
                )
            buffer_gaussian = conduct_decompress(
                args,
                model_path,
                args.init_3dgs,
                motion_save_path,
                motion_prior_save_path,
                mask_save_path,
                ply_save_path,
                use_buffer=True,
                buffer_gaussian=buffer_gaussian,
                origin_motion_xyz=origin_motion_xyz,
                json_path=json_path,
            )
    logger.save_general_info() 

def validate(gaussians, scene, args, save_path='', save_img=False, logger=None, json_path='rendering_info.json'):
        
    logger.log_info("Starting validation...")

    device = torch.device(f"cuda:{args.gpu}")
    lpips_model = get_lpips_fn(device)
    views = scene.getTestCameras()
    # views = scene.getTrainCameras()
    with torch.no_grad():
        ssim_test = {}
        L1_test = {}
        lpips_test = {}
        psnr_test = {}
        for id, view in enumerate(tqdm.tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipe=args, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))["render"]  # [3, H, W]
            if save_img:
                torchvision.utils.save_image(rendering, os.path.join(save_path, f"{id}.png"))
            gt = view.original_image[0:3, :, :].to(device)
            rendering = torch.round(rendering.mul(255).clamp_(0, 255)) / 255.0
            ssim_test[id] = (ssim(rendering, gt)).mean().double().item()
            L1_test[id] = l1_loss(rendering, gt).mean().double().item()
            lpips_test[id] = lpips_model(rendering, gt).mean().double().item()
            psnr_test[id] = psnr(rendering, gt).mean().double().item()

            ssim_loss = ssim(rendering, gt).item()
            L1_loss = l1_loss(rendering, gt).item()

            # logger.log_info(f"View {id}: PSNR={psnr_test[id]:.4f}, SSIM={ssim_test[id]:.4f}, LPIPS={lpips_test[id]:.4f}, L1={L1_test[id]:.4f}")

            torch.cuda.empty_cache()

        ssim_avg = sum(ssim_test.values()) / len(views)
        Ll1_avg = sum(L1_test.values()) / len(views)
        lpips_avg = sum(lpips_test.values()) / len(views)
        psnr_avg = sum(psnr_test.values()) / len(views)

        logger.log_eval({
            'PSNR': psnr_avg,
            'SSIM': ssim_avg,
            'LPIPS': lpips_avg,
            'L1': Ll1_avg
        })

    with open(os.path.join(save_path, json_path), 'w') as f:
        data = {'PSNR': psnr_avg,
                'SSIM': ssim_avg,
                'LPIPS': lpips_avg,
                'L1': Ll1_avg
                }
        
        details = {
            'PSNR': psnr_test,
            'SSIM': ssim_test,
            'LPIPS': lpips_test,
            'L1': L1_test
            }
        
        final_data = {'average': data, 'details': details}

        json.dump(final_data, f, indent=True)
    
    logger.add_to_list('psnr', psnr_avg)
    logger.add_to_list('ssim', ssim_avg)
    logger.add_to_list('lpips', lpips_avg)

def arg_parse():
    parser = argparse.ArgumentParser(description='FCGS_D')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--gpu', type=int, default=0)

    # hyper parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lambda_size', type=float, default=1e-3, help='scaler of z')
    parser.add_argument('--Q_y', type=float, default=1, help='granularity of quantization for latent code')
    parser.add_argument('--Q_z', type=float, default=1, help='granularity of quantization for hyper latent code')
    parser.add_argument('--gof_size', type=int, default=5, help='number of gaussian frames in a group')
    parser.add_argument('--gaussian_feature_dim', type=int, default=56, help='dimension of gaussian feature')
    parser.add_argument('--motion_dim', type=int, default=4, help='dimension of motion')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden layers of MLP')
    parser.add_argument('--lat_dim', type=int, default=128, help='dimension of latent code')
    parser.add_argument('--downsample_rate', type=int, default=100, help='downsample rate')
    parser.add_argument('--knn_num', type=int, default=10, help='number of nearest neighbors')
    parser.add_argument('--max_point_num', type=int, default=1500, help='maximum number of control points for motion estimation')
    parser.add_argument('--motion_limit', type=float, default=0.1, help='motion limit')

    # path setting
    parser.add_argument('--init_3dgs', type=str, default=None, help='path to initial 3dgs')
    parser.add_argument('--next_3dgs', type=str, default=None, help='path to next 3dgs')
    parser.add_argument('--motion_estimator_path', type=str, default=None, help='path to motion estimator')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to checkpoint pth file') 
    parser.add_argument('--dataset_path', type=str, default='./data_video', help='path to dataset') 
    parser.add_argument('--frame_path', type=str, default=None, help='path to current working frame') 
    
    # training/testing config
    parser.add_argument('--conduct_training', action='store_true', help='conduct training')
    parser.add_argument('--conduct_compress', action='store_true', help='conduct compress')
    parser.add_argument('--conduct_decompress', action='store_true', help='conduct decompress')
    parser.add_argument('--use_first_as_test', action='store_true', help='use first frame as test, same as 3DGStream')
    parser.add_argument('--random_init', action='store_true', help='randomly initialize the I frame')
    parser.add_argument('--without_refinement', action='store_true', help='without refinement')
    parser.add_argument('--without_context', action='store_true', help='without contest prior')
    parser.add_argument('--without_hyper', action='store_true', help='without hyperprioe')
    parser.add_argument('--per_frame', action='store_true', help='train per frame')
    parser.add_argument('--joint', action='store_true', help='train jointly')
    parser.add_argument('--use_scenes', action='store_true', help='use scenes')
    parser.add_argument('--use_gof', action='store_true', help='use group of frames')
    parser.add_argument('--add_position_noise', action='store_true', help='add position noise to the initial 3dgs')

    parser.add_argument('--frame_start', type=int, default=1, help='start frame for training')
    parser.add_argument('--frame_end', type=int, default=300, help='end frame for training')
    parser.add_argument('--test_frame_start', type=int, default=1)
    parser.add_argument('--test_frame_end', type=int, default=300)

    parser.add_argument('--dynamicGS_type', type=str, default='control_point', choices=['3dgstream_explicit', '3dgstream_implicit', 'control_point'], help='type of dynamic Gaussian Splatting')
    parser.add_argument('--dataset', type=str, default='dynerf')
    parser.add_argument('--dataset_list', nargs='+', default=['dynerf'])
    parser.add_argument('--scene_list', nargs='+', default=['sear_steak'])
    parser.add_argument('--checkpoint_iteration', type=int, default=1000, help='checkpoint saving iteration')
    parser.add_argument('--logger', type=Logger, default=None)
    parser.add_argument('--lmd', type=float, default=16e-4, help='lmd value for training')
    parser.add_argument('--use_added', action='store_true', help='use added for training')
    parser.add_argument('--use_I_compress', action='store_true', help='use I frame compressing for GoF training')

    parser.add_argument(
        '--gpcc_codec_path',
        type=str,
        default='/mnt/data3/ctx/mpeg-pcc-tmc13/build/tmc3/tmc3',
        help='Path to GPCC codec executable (tmc3).',
    )

    parser.add_argument(
        '--zstd_bin',
        type=str,
        default='zstd',
        help='zstd executable (e.g. zstd or /usr/bin/zstd).',
    )
    parser.add_argument(
        '--zstd_level',
        type=int,
        default=22,
        help='zstd compression level (1-22). Levels > 19 require zstd --ultra.',
    )

    parser.add_argument(
        '--zstd_long',
        type=int,
        default=31,
        help='zstd --long window log (e.g. 27-31). Larger can improve ratio but increases memory/time.',
    )

    parser.add_argument(
        '--zstd_threads',
        type=int,
        default=0,
        help='zstd threads (0 => all cores).',
    )

    parser.add_argument(
        '--motion_xyz_quant',
        type=str,
        default='int16',
        choices=['none', 'f16', 'int16', 'int8', 'int16_dz', 'int8_dz'],
        help='Quantization mode for origin_motion_xyz before zstd. int16 usually compresses best.',
    )
    parser.add_argument(
        '--motion_xyz_quant_step',
        type=float,
        default=1e-4,
        help='Quantization step for int16 mode (smaller => higher quality, larger => smaller size).',
    )
    parser.add_argument(
        '--motion_xyz_deadzone',
        type=float,
        default=None,
        help='Deadzone size for *_dz modes (default: equals quant_step). Values within deadzone are forced to 0.',
    )
    parser.add_argument(
        '--motion_xyz_quant_bias',
        type=str,
        default='mean',
        choices=['zero', 'mean'],
        help='Bias mode for quantization. mean often improves compression ratio.',
    )
    parser.add_argument(
        '--motion_xyz_stats',
        action='store_true',
        help='Print distribution statistics for origin_motion_xyz (min/max/mean/std, |x| percentiles, near-zero mass).',
    )

    args = parser.parse_args()

    logger = Logger(args.model_path)
    logger.log_args(args)
    args.logger = logger

    return args

def main(args):
    if args.conduct_training: # conduct training
        train_frame_gof(args)
    
    if args.conduct_compress and args.conduct_decompress: # compress and decompress as GoF
        for scene_name in args.scene_list:
            args.scene_name = scene_name
            conduct_compress_decompress_as_gof(args, args.test_frame_start, args.test_frame_end, json_path='rendering_info.json')
            # debug_control_point(args, args.test_frame_start, args.test_frame_end)
    
    # if  args.conduct_decompress: # decompress video
    #     for scene_name in args.scene_list:
    #         args.scene_name = scene_name
    #         conduct_decompress_video(args, args.test_frame_start, args.test_frame_end , json_path = 'rendering_info_decompress.json')

if __name__ == "__main__":

    args = arg_parse()

    torch.cuda.set_device(args.gpu)

    main(args)
        
