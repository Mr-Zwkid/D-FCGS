import argparse
import glob
import os
import shutil
import sys
from typing import Optional


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _require_cuda_or_explain(context: str) -> None:
    """Fail fast with a helpful message if CUDA runtime is unavailable.

    This pipeline (FCGS + 3DGStream) requires a working NVIDIA driver + CUDA runtime.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "当前 Python 环境里找不到 torch。\n"
            "请确认你是在装有 PyTorch 的环境里运行（例如先 `conda activate dfcgs` 再运行脚本）。\n"
            f"原始错误: {type(exc).__name__}: {exc}"
        ) from exc

    # IMPORTANT:
    # On some systems NVML initialization fails (or a single faulty GPU breaks enumeration),
    # causing torch.cuda.is_available()/device_count() to return False even though CUDA kernels
    # can run on a selected device. Here we validate CUDA by actually allocating on cuda:0.
    try:
        _ = torch.tensor([0.0], device="cuda")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{context}: CUDA 初始化/分配失败（无法在 cuda:0 上创建 tensor）。\n"
            "这通常不是代码问题，而是运行环境问题：\n"
            "- NVIDIA 驱动异常/版本不匹配\n"
            "- 你在没有 GPU 的节点上运行\n"
            "- 容器未做 GPU passthrough（/dev/nvidia* 不存在）\n"
            "- 机器上存在坏卡导致初始化失败：可尝试用 `--cuda_visible_devices 0` 屏蔽\n\n"
            "建议你先检查：\n"
            "1) `nvidia-smi`\n"
            "2) `echo $CUDA_VISIBLE_DEVICES`\n"
            "3) `CUDA_VISIBLE_DEVICES=0 python - <<'PY'\nimport torch\nprint(torch.__version__)\nprint(torch.version.cuda)\nprint(torch.tensor([1.], device='cuda'))\nPY`\n\n"
            f"原始错误: {type(exc).__name__}: {exc}"
        ) from exc


def _ensure_sys_path():
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def _latest_point_cloud_ply(model_dir: str, prefer_iteration: Optional[int] = None) -> str:
    """Resolve point_cloud.ply under a 3DGStream model directory.

    model_dir is expected like: .../frameXXXXXX/gs
    """
    if prefer_iteration is not None:
        candidate = os.path.join(model_dir, "point_cloud", f"iteration_{prefer_iteration}", "point_cloud.ply")
        if os.path.exists(candidate):
            return candidate

    pattern = os.path.join(model_dir, "point_cloud", "iteration_*", "point_cloud.ply")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No point_cloud.ply found under: {model_dir}")
    return matches[-1]


def _write_recon_as_3dgstream_model(recon_ply_path: str, out_model_dir: str, iteration: int, overwrite: bool) -> None:
    dst = os.path.join(out_model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if os.path.exists(dst) and not overwrite:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(recon_ply_path, dst)


def _ensure_ply_linked_as_3dgstream_model(ply_path: str, model_dir: str, iteration: int, overwrite: bool) -> None:
    """Ensure 3DGStream can load a ply by placing point_cloud.ply under model_dir.

    3DGStream expects: {model_dir}/point_cloud/iteration_{iteration}/point_cloud.ply
    We prefer a symlink to avoid duplicating/accidentally using a stale copy.
    """
    dst = os.path.join(model_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    # If already correct and overwrite is off, keep it.
    if os.path.exists(dst) and not overwrite:
        try:
            if os.path.samefile(dst, ply_path):
                return
        except Exception:
            return

    if os.path.lexists(dst):
        os.remove(dst)

    # Use relative symlink for portability inside the dataset tree.
    try:
        rel_target = os.path.relpath(ply_path, start=os.path.dirname(dst))
        os.symlink(rel_target, dst)
    except Exception:
        # Fall back to copying if symlink is not permitted.
        shutil.copy2(ply_path, dst)


def _fcgs_checkpoint_path(lmd_str: str, ckpt_path: Optional[str], ckpt_dir: Optional[str]) -> str:
    if ckpt_path is not None:
        return ckpt_path
    if ckpt_dir is not None:
        return os.path.join(ckpt_dir, f"checkpoint_{lmd_str}.pkl")
    return f"/mnt/data3/ctx/FCGS/checkpoints/checkpoint_{lmd_str}.pkl"


def fcgs_compress_ply(ply_path_from: str, bit_path_to: str, lmd_str: str, ckpt_path: str, determ: int = 1, nr: int = 3) -> None:
    _ensure_sys_path()
    import time
    import numpy as np
    import torch
    from scene import GaussianModel
    from fcgs.FCGS_model import FCGS

    # NOTE: We still require CUDA for FCGS, but loading ply itself can be kept on CPU.
    with torch.no_grad():
        gaussians = GaussianModel(3)
        gaussians.load_ply(path=ply_path_from, device="cpu")

    g_xyz = gaussians._xyz.detach()
    num_gaussians = g_xyz.shape[0]

    per_step_size = 1_000_000
    if 1_000_000 < num_gaussians < 1_100_000:
        per_step_size = 1_100_000

    features_dc = gaussians._features_dc.detach().view(num_gaussians, -1)
    features_rest = gaussians._features_rest.detach().view(num_gaussians, -1)
    opacity = gaussians._opacity.detach()
    scaling = gaussians._scaling.detach()
    rotation = gaussians._rotation.detach()
    g_fea = torch.cat([features_dc, features_rest, opacity, scaling, rotation], dim=-1)

    finite_max = torch.where(torch.isinf(g_fea), torch.full_like(g_fea, -torch.inf), g_fea).max(0, keepdim=True)[0]
    g_fea = torch.where(torch.isinf(g_fea), finite_max, g_fea)

    step_num = int(np.ceil(num_gaussians / per_step_size))
    chunk_size_list = [2_000_000, 1_000_000, 1_000_000]

    cm = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=nr,
    ).cuda()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"FCGS checkpoint not found: {ckpt_path}")
    cm.load_state_dict(torch.load(ckpt_path), strict=True)
    cm.eval()

    os.makedirs(bit_path_to, exist_ok=True)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(bit_path_to, f"{lmd_str}/{s}")
            os.makedirs(bit_save_path, exist_ok=True)
            g_xyz_in = g_xyz[s * per_step_size : s * per_step_size + per_step_size].to(device="cuda", non_blocking=True)
            g_fea_in = g_fea[s * per_step_size : s * per_step_size + per_step_size].to(device="cuda", non_blocking=True)
            cm.compress(g_xyz_in, g_fea_in, root_path=bit_save_path, chunk_size_list=chunk_size_list, determ_codec=determ)
    torch.cuda.synchronize()
    _ = time.time() - t1


def fcgs_decompress_ply(ply_path_to: str, bit_path_from: str, lmd_str: str, ckpt_path: str, determ: int = 1, nr: int = 3) -> None:
    _ensure_sys_path()
    import os
    import torch
    import torch.nn as nn
    from scene import GaussianModel
    from fcgs.FCGS_model import FCGS

    lmd_dir = os.path.join(bit_path_from, str(lmd_str))
    if not os.path.exists(lmd_dir):
        raise FileNotFoundError(f"FCGS bitstream folder not found: {lmd_dir}")

    step_num = len([d for d in os.listdir(lmd_dir) if os.path.isdir(os.path.join(lmd_dir, d))])
    chunk_size_list = [2_000_000, 1_000_000, 1_000_000]

    cm = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=nr,
    ).cuda()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"FCGS checkpoint not found: {ckpt_path}")
    cm.load_state_dict(torch.load(ckpt_path), strict=True)
    cm.eval()

    g_xyz_list = []
    g_fea_list = []
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(bit_path_from, f"{lmd_str}/{s}")
            g_xyz_out, g_fea_out = cm.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list)
            g_xyz_list.append(g_xyz_out)
            g_fea_list.append(g_fea_out)

    g_xyz = torch.cat(g_xyz_list, dim=0)
    g_fea = torch.cat(g_fea_list, dim=0)
    f_dc, f_rst, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3, 45, 1, 3, 4], dim=-1)

    gaussians = GaussianModel(3)
    gaussians._xyz = nn.Parameter(g_xyz)
    gaussians._features_dc = nn.Parameter(f_dc.view(-1, 1, 3))
    gaussians._features_rest = nn.Parameter(f_rst.view(-1, 15, 3))
    gaussians._opacity = nn.Parameter(op.view(-1, 1))
    gaussians._scaling = nn.Parameter(sc.view(-1, 3))
    gaussians._rotation = nn.Parameter(ro.view(-1, 4))

    os.makedirs(os.path.dirname(ply_path_to), exist_ok=True)
    gaussians.save_ply(ply_path_to)


def _run_3dgstream_train_frames(gs_repo_dir: str, scene_dir: str, scene: str, model_path: str, frame_start: int, frame_end_inclusive: int, first_load_iteration: int, load_iteration: int) -> int:
    """Run 3DGStream/train_frames.py for frames [frame_start, frame_end_inclusive]."""
    gs_dir = os.path.join(gs_repo_dir, "3DGStream")
    train_frames_py = os.path.join(gs_dir, "train_frames.py")
    cfg_path = os.path.join(gs_dir, "configs", "cfg_args.json")
    ntc_conf_path = os.path.join(gs_dir, "configs", "cache", "cache_F_4.json")
    ntc_path = os.path.join(gs_dir, "ntc", f"{scene}_ntc_params_F_4.pth")

    scene_dir_abs = os.path.abspath(scene_dir)
    model_path_abs = os.path.abspath(model_path)

    # train_frames uses Python slicing frames[frame_start:frame_end], so end is exclusive.
    frame_end_exclusive = frame_end_inclusive + 1
    # NOTE: cache_warmup uses relative paths like ./configs/cache/cache_F_4.json.
    # We run inside 3DGStream/ to make config-relative paths work.
    cmd = (
        f"cd {gs_dir} && "
        f"python {train_frames_py} "
        f"--read_config --config_path {cfg_path} "
        f"--ntc_conf_path {ntc_conf_path} "
        f"-o {scene_dir_abs} "
        f"-m {model_path_abs} "
        f"-v {scene_dir_abs} "
        f"--image images "
        f"--first_load_iteration {first_load_iteration} "
        f"--load_iteration {load_iteration} "
        f"--ntc_path {ntc_path} "
        f"--frame_start {frame_start} "
        f"--frame_end {frame_end_exclusive} "
        f"--sh_degree 3 --eval"
    )
    return os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Derive 3DGStream frames with periodic FCGS compress/decompress on keyframes.")
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Optional: set CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1,2') to mask faulty GPUs before importing torch.",
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Base directory containing multi-camera videos.")
    parser.add_argument("--scene_list", nargs="*", default=None, help="List of scene names to process.")
    parser.add_argument("--start_frame", type=int, default=1, help="Start frame index (inclusive).")
    parser.add_argument("--end_frame", type=int, default=299, help="End frame index (inclusive).")
    parser.add_argument(
        "--gof_size",
        type=int,
        default=10,
        help="GoF size (e.g. 10 => keyframes at 0,10,20,...; trains frames key+1..key+9).",
    )
    parser.add_argument("--first_load_iteration", type=int, default=4000, help="3DGStream iteration to load for frame000000.")
    parser.add_argument("--load_iteration", type=int, default=150, help="3DGStream iteration to load for non-first frames.")

    parser.add_argument("--fcgs_lmd", type=str, default="0.0016", help="FCGS lambda id (used in checkpoint filename checkpoint_{lmd}.pkl).")
    parser.add_argument("--fcgs_ckpt", type=str, default=None, help="Path to FCGS checkpoint .pkl. If set, overrides --fcgs_ckpt_dir.")
    parser.add_argument("--fcgs_ckpt_dir", type=str, default=None, help="Directory containing FCGS checkpoints (checkpoint_{lmd}.pkl).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing FCGS recon/model artifacts.")
    parser.add_argument("--skip_fcgs_if_exists", action="store_true", help="If recon ply exists, skip FCGS compress/decompress.")
    parser.add_argument("--determ", type=int, default=1, help="FCGS determ codec flag.")
    parser.add_argument("--nr", type=int, default=3, help="FCGS norm radius.")

    args = parser.parse_args()

    # Some machines have one faulty GPU that breaks cudaGetDeviceCount().
    # Allow masking GPUs before any torch import happens.
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        # Make device indexing stable.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # # Fail fast with a clear error if CUDA runtime isn't usable.
    # _require_cuda_or_explain(context="启动预处理流水线")

    dataset_dir = args.dataset_dir
    scenes = os.listdir(dataset_dir) if args.scene_list is None else args.scene_list
    if args.gof_size <= 0:
        raise ValueError("--gof_size must be positive")
    if args.start_frame < 1:
        raise ValueError("--start_frame should be >= 1 (frame000000 is the initial keyframe)")

    repo_root = _repo_root()
    ckpt_path = _fcgs_checkpoint_path(args.fcgs_lmd, args.fcgs_ckpt, args.fcgs_ckpt_dir)

    for scene in scenes:
        scene_dir = os.path.join(dataset_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        # First keyframe is the greatest multiple of gof_size that is < start_frame.
        keyframe = ((args.start_frame - 1) // args.gof_size) * args.gof_size
        while keyframe <= args.end_frame:
            group_start = max(args.start_frame, keyframe + 1)
            # For GoF size N, keyframes are spaced by N and we train N-1 inter frames.
            group_end = min(args.end_frame, keyframe + args.gof_size - 1)
            if group_start > group_end:
                # No inter frames left to train.
                break

            # 1) Prefer using an already reconstructed FCGS ply as keyframe (I-frame).
            # In many datasets, frame000000/frame000010/... already contain: FCGS_point_cloud_{lmd}.ply.
            fcgs_bit_dir = os.path.join(scene_dir, f"frame{keyframe:06d}", f"FCGS_bitstreams_{args.fcgs_lmd}")
            fcgs_recon_ply = os.path.join(scene_dir, f"frame{keyframe:06d}", f"FCGS_point_cloud_{args.fcgs_lmd}.ply")
            if not (os.path.exists(fcgs_recon_ply) and not args.overwrite):
                # 2) If recon ply doesn't exist (or overwrite requested), derive it via FCGS.
                keyframe_gs_dir = os.path.join(scene_dir, f"frame{keyframe:06d}", "gs")
                prefer_iter = args.first_load_iteration if keyframe == 0 else args.load_iteration
                keyframe_ply = _latest_point_cloud_ply(keyframe_gs_dir, prefer_iteration=prefer_iter)

                _require_cuda_or_explain(context=f"FCGS 压缩/解压关键帧 frame{keyframe:06d}")
                if args.overwrite and os.path.exists(fcgs_bit_dir):
                    shutil.rmtree(fcgs_bit_dir)
                fcgs_compress_ply(
                    ply_path_from=keyframe_ply,
                    bit_path_to=fcgs_bit_dir,
                    lmd_str=args.fcgs_lmd,
                    ckpt_path=ckpt_path,
                    determ=args.determ,
                    nr=args.nr,
                )
                fcgs_decompress_ply(
                    ply_path_to=fcgs_recon_ply,
                    bit_path_from=fcgs_bit_dir,
                    lmd_str=args.fcgs_lmd,
                    ckpt_path=ckpt_path,
                    determ=args.determ,
                    nr=args.nr,
                )

            # 3) Make 3DGStream load the keyframe point cloud directly from
            #    frameXXXXXX/FCGS_point_cloud_{lmd}.ply by linking it into the expected model structure.
            keyframe_frame_dir = os.path.join(scene_dir, f"frame{keyframe:06d}")
            recon_model_dir = keyframe_frame_dir
            recon_iter = args.first_load_iteration if keyframe == 0 else args.load_iteration
            _ensure_ply_linked_as_3dgstream_model(fcgs_recon_ply, recon_model_dir, iteration=recon_iter, overwrite=args.overwrite)

            # 4) Run 3DGStream for this GoF segment.
            ret = _run_3dgstream_train_frames(
                gs_repo_dir=repo_root,
                scene_dir=scene_dir,
                scene=scene,
                model_path=recon_model_dir,
                frame_start=group_start,
                frame_end_inclusive=group_end,
                first_load_iteration=args.first_load_iteration,
                load_iteration=args.load_iteration,
            )
            if ret != 0:
                raise RuntimeError(f"3DGStream failed for scene={scene}, frames={group_start}-{group_end}, exit={ret}")

            # Next keyframe is spaced by gof_size (e.g. 0->10->20...).
            keyframe = keyframe + args.gof_size


if __name__ == "__main__":
    main()