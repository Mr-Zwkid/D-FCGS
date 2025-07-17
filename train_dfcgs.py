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
from model.FCGS_D_model import FCGS_D

from gaussian_renderer import render
import lpips
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

from utils.logger import Logger
import glob

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

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
        base_dir = f'{args.dataset_path}/{args.dataset}/{args.scene_name}'
        if frame_cur >= 0:
            init_pattern = f'{base_dir}/frame{frame_cur:06d}/gs/point_cloud/iteration_*/point_cloud.ply'
            args.init_3dgs = path_match(init_pattern)

        frame_next = frame_cur + 1 if frame_next is None else frame_next
        next_pattern = f'{base_dir}/frame{frame_next:06d}/gs/point_cloud/iteration_*/point_cloud.ply'
        args.next_3dgs = path_match(next_pattern)
        
        args.source_path = f'{base_dir}/frame{frame_next:06d}' # the source path for the next frame images, used for evaluation
        args.frame_path = os.path.join(args.model_path, args.scene_name, f'frame{frame_next:06d}')
    else:
        raise ValueError(f"Unknown dynamicGS_type: {args.dynamicGS_type}")

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
        
        loss_render, loss_size = model()

        lambda_size = args.lambda_size # if i > args.iterations // 3 else 0.0
        loss = loss_render + loss_size * lambda_size

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
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        }, additional_info={'frame_cur': frame_cur, 'frame_next': frame_next})

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        # save checkpoint
        if args.checkpoint is not None and i % args.checkpoint == 0:
            model_path = os.path.join(args.model_path.split('frame')[0], f'checkpoint_{i}.pth')
            logger.log_info(f"Saving checkpoint to {model_path}")
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.log_checkpoint(model_path)

        # update buffer frame
        with torch.no_grad():
            buffer_gaussian = model.buffer_capture()

    args.model_path = args.model_path.split('frame')[0]
    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.log_info(f"Saved final model to {model_path}")

    loss_curve_path = os.path.join(args.model_path, 'loss_curve.png')
    plot_loss_curve(render_loss, size_loss, total_loss, loss_curve_path)
    logger.log_info(f"Saved loss curves to {loss_curve_path}")

    logger.log_training_complete()

def conduct_compress_decompress_as_gof(args, frame_start, frame_end):
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
            logger.log_info(f"Compressing frame {frame} as I frame (Currently Skipping)...")
        elif frame % args.gof_size == 1:
            conduct_compress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, args.next_3dgs)
            buffer_gaussian = conduct_decompress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, ply_save_path)
        else:
            conduct_compress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, args.next_3dgs, use_buffer=True, buffer_gaussian=buffer_gaussian)
            conduct_decompress(args, model_path, args.init_3dgs, motion_save_path, motion_prior_save_path, mask_save_path, ply_save_path, buffer_gaussian=buffer_gaussian)
    logger.save_general_info()
            
def conduct_compress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', nxt_gaussians_path=None, use_buffer=False, buffer_gaussian=None):
    logger = args.logger
    logger.log_info(f"Model from {model_path}")
    logger.log_info(f"Compressing {nxt_gaussians_path} with reference of {init_3dgs_path}")
    
    with torch.no_grad():
        model = torch.load(model_path, map_location='cuda:0')
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()

        res, duration = myFCGS_D.compress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path, mask_path, dynamicGS_type=args.dynamicGS_type, nxt_gaussians_path=nxt_gaussians_path, use_buffer=use_buffer, buffer_gaussian=buffer_gaussian, add_position_noise=args.add_position_noise)
        logger.add_to_list('compress_time', duration)
        logger.log_info(f"Compression time: {duration:.2f} seconds")
        with open(os.path.join(args.frame_path, 'size.json'), 'w') as f:
            json.dump(res, f, indent=True)
        logger.add_to_list('total_size', res['bits_total'])
        
        logger.log_info(f'Compression Result: {res}') 

def conduct_decompress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', save_ply_path = './output.ply', buffer_gaussian=None):
    logger = args.logger
    logger.log_info(f"Decompressing next Gaussian Frame from {init_3dgs_path}")
    
    with torch.no_grad():
        model = torch.load(model_path, map_location='cuda:0')
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene

        rec_gaussians, duration = myFCGS_D.decompress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path, mask_path, dynamicGS_type=args.dynamicGS_type, buffer_gaussian=buffer_gaussian)
        logger.add_to_list('decompress_time', duration)
        logger.log_info(f"Decompression time: {duration:.2f} seconds")
        # rec_gaussians.save_ply(save_ply_path)
        validate(rec_gaussians, scene, args,  args.frame_path, save_img=True, logger=logger)
    return rec_gaussians
        
def validate(gaussians, scene, args, save_path='', save_img=False, logger=None):
        
    logger.log_info("Starting validation...")
    
    # lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
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
            gt = view.original_image[0:3, :, :].to("cuda")
            rendering = torch.round(rendering.mul(255).clamp_(0, 255)) / 255.0
            ssim_test[id] = (ssim(rendering, gt)).mean().double().item()
            L1_test[id] = l1_loss(rendering, gt).mean().double().item()
            lpips_test[id] = lpips_fn(rendering, gt).mean().double().item()
            psnr_test[id] = psnr(rendering, gt).mean().double().item()

            ssim_loss = ssim(rendering, gt).item()
            L1_loss = l1_loss(rendering, gt).item()

            logger.log_info(f"View {id}: PSNR={psnr_test[id]:.4f}, SSIM={ssim_test[id]:.4f}, LPIPS={lpips_test[id]:.4f}, L1={L1_test[id]:.4f}")

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

    with open(os.path.join(save_path, 'rendering_info.json'), 'w') as f:
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
    parser = argparse.ArgumentParser(description='D-FCGS Official Implementation')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    # hyper parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lambda_size', type=float, default=1e-3, help='weight for rate loss')
    parser.add_argument('--Q_y', type=float, default=1, help='granularity of quantization for latent code')
    parser.add_argument('--Q_z', type=float, default=1, help='granularity of quantization for hyper latent code')
    parser.add_argument('--gof_size', type=int, default=5, help='number of gaussian frames in a group')
    parser.add_argument('--gaussian_feature_dim', type=int, default=56, help='dimension of gaussian feature')
    parser.add_argument('--motion_dim', type=int, default=7, help='dimension of motion')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden layers of MLP')
    parser.add_argument('--lat_dim', type=int, default=16, help='dimension of latent code')
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
    parser.add_argument('--use_first_as_test', action='store_true', help='use first frame as test, the same as 3DGStream')
    parser.add_argument('--random_init', action='store_true', help='randomly initialize the I frame')
    parser.add_argument('--without_refinement', action='store_true', help='ablation without refinement')
    parser.add_argument('--without_context', action='store_true', help='ablation without context prior')
    parser.add_argument('--without_hyper', action='store_true', help='ablation without hyperprior')
    parser.add_argument('--add_position_noise', action='store_true', help='add position noise to the initial 3dgs')

    parser.add_argument('--frame_start', type=int, default=1, help='start frame for training')
    parser.add_argument('--frame_end', type=int, default=300, help='end frame for training')
    parser.add_argument('--test_frame_start', type=int, default=1)
    parser.add_argument('--test_frame_end', type=int, default=300)

    parser.add_argument('--dynamicGS_type', type=str, default='control_point', choices=['3dgstream_explicit', '3dgstream_implicit', 'control_point'], help='type of dynamic Gaussian Splatting')
    parser.add_argument('--dataset', type=str, default='dynerf')
    parser.add_argument('--dataset_list', nargs='+', default=['dynerf'])
    parser.add_argument('--scene_list', nargs='+', default=['sear_steak'])
    parser.add_argument('--checkpoint_iteration', type=int, default=None, help='checkpoint saving iteration')
    parser.add_argument('--logger', type=Logger, default=None)

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
            conduct_compress_decompress_as_gof(args, args.test_frame_start, args.test_frame_end)
    
if __name__ == "__main__":

    args = arg_parse()

    torch.cuda.set_device(args.gpu)

    main(args)
