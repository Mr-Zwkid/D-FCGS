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

def train_frame_setup(args, frame_cur, frame_next=None):
    # train scenes jointly

    if args.dynamicGS_type == '3dgstream':
        if frame_cur == 0:
            args.init_3dgs = f'/SSD2/chenzx/Projects/Dataset4Compression/init_3dgs/{args.dataset}/{args.scene_name}/frame000000/point_cloud/iteration_4000/point_cloud.ply'
            
            args.motion_estimator_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.dataset}/{args.scene_name}/frame000001/NTC.pth'
        else:
            args.init_3dgs = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.dataset}/{args.scene_name}/frame{frame_cur:06d}/point_cloud/iteration_150/point_cloud.ply'
            args.motion_estimator_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.dataset}/{args.scene_name}/frame{frame_cur+1:06d}/NTC.pth'
    
        args.source_path = f'/SDD_D/zwk/data_dynamic/{args.dataset}/{args.scene_name.split("-")[0]}/frame{frame_cur+1:06d}'
        args.model_path = os.path.join(args.model_path.split('frame')[0], f'frame{frame_cur+1:06d}')
    
    elif args.dynamicGS_type == '3dgstream_explicit' or args.dynamicGS_type == '3dgstream_implicit':
        if frame_cur == 0:
            args.init_3dgs = f'/SSD2/chenzx/Projects/Dataset4Compression/init_3dgs/{args.dataset}/{args.scene_name}/frame000000/point_cloud/iteration_4000/point_cloud.ply'
        else:
            args.init_3dgs = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.dataset}/{args.scene_name}/frame{frame_cur:06d}/point_cloud/iteration_150/point_cloud.ply'
        
        frame_next = frame_cur + 1 if frame_next is None else frame_next
        args.next_3dgs = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.dataset}/{args.scene_name}/frame{frame_next:06d}/point_cloud/iteration_150/point_cloud.ply'
        
        base_dir = '/SDD_D/zwk/data_dynamic' if args.dataset == 'dynerf' or args.dataset == 'meetroom' else '/SSD2/chenzx/Projects/Dataset4Compression'
        args.source_path = f'{base_dir}/{args.dataset}/{args.scene_name.split("-")[0]}/frame{frame_next:06d}'
        args.model_path = os.path.join(args.model_path.split('frame')[0], f'frame{frame_next:06d}')
      
def train_frame(args):
    logger = args.logger

    for frame in range(args.frame_start, args.frame_end):
        train_frame_setup(args, frame)

        if args.conduct_training:
            # logger.logger.info(f"\n[Training frame {frame}]")
            print(f"\n[Training frame {frame}]")
            train(args)

        if args.conduct_compress:
            logger.logger.info(f"\n[Compressing frame {frame}]")
            print(f"\n[Compressing frame {frame}]")
            conduct_compress(args, os.path.join(args.model_path, 'model.pth'), args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'))
        
        if args.conduct_decompress:
            logger.logger.info(f"\n[Decompressing frame {frame}]")
            print(f"\n[Decompressing frame {frame}]")
            conduct_decompress(args, os.path.join(args.model_path, 'model.pth'), args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), os.path.join(args.model_path, 'output.ply'))
        
        logger.logger.info('\n')

def train_frame_gof(args):
    logger = args.logger
    logger.logger.info("Starting training within gof...")

    model = FCGS_D(args).cuda()

    # checkpoint
    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        logger.logger.info(f"Loaded checkpoint from {args.checkpoint_path}")
    
    model.train()

    render_loss = []
    size_loss = []
    mask_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")
    buffer_gaussian = None

    for i in pbar:
        with torch.no_grad():
            if args.use_scenes and i % args.gof_size == 0:
                torch.cuda.empty_cache()
                args.scene_name = args.scene_list[torch.randint(0, len(args.scene_list), (1,)).item()]
                print(f"Scene: {args.scene_name}")

                frame_cur = torch.randint(args.frame_start, args.frame_end, (1,)).item() // args.gof_size * args.gof_size
                frame_next = frame_cur + 1
                train_frame_setup(args, frame_cur, frame_next)
                model.refresh_settings(args)
            else:
                torch.cuda.empty_cache()
                frame_cur = min(frame_cur + 1, args.frame_end - 2)
                frame_next = min(frame_next + 1, args.frame_end - 1)
                train_frame_setup(args, frame_cur, frame_next)
                model.buffer_loading(args, buffer_gaussian)
        print(f"Frame: {frame_cur} -> {frame_next}")
        print('Scene:', args.scene_name)

        loss_render, size, loss_mask, loss_motion = model()
        loss_size = size

        lambda_size = args.lambda_size if i > args.iterations // 3 else 0.0
        # lambda_motion = 0.01 if i > args.iterations // 3 else 0.0
        lambda_motion = 0.01
        # lambda_render = 1 if i > args.iterations // 10 else 0.0
        # lambda_size = args.lambda_size
        lambda_render = 1
        loss = loss_render * lambda_render + loss_size * lambda_size + loss_mask * 0.01 + loss_motion *lambda_motion

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        torch.cuda.empty_cache()

        render_loss.append(loss_render.item())
        size_loss.append(loss_size.item())
        mask_loss.append(loss_mask.item())
        total_loss.append(loss.item())

        # 日志记录
        logger.log_iteration(i, {
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        }, additional_info={'frame_cur': frame_cur, 'frame_next': frame_next})

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        if args.checkpoint is not None and int(i) == args.checkpoint:
            model_path = os.path.join(args.model_path.split('frame')[0], f'checkpoint_{args.checkpoint}.pth')
            logger.logger.info(f"Saving checkpoint to {model_path}")
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.log_checkpoint(model_path)
        with torch.no_grad():
            buffer_gaussian = model.buffer_capture()

    args.model_path = args.model_path.split('frame')[0]
    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.logger.info(f"Saved final model to {model_path}")

    loss_curve_path = os.path.join(args.model_path, 'loss_curve.png')
    plot_loss_curve(render_loss, size_loss, total_loss, loss_curve_path)
    logger.logger.info(f"Saved loss curves to {loss_curve_path}")

    logger.log_training_complete()

def train_frame_jointly_stepping(args):

    logger = args.logger
    logger.logger.info("Starting joint training with stepping...")
    
    model = FCGS_D(args).cuda()

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        logger.logger.info(f"Loaded checkpoint from {args.checkpoint_path}")

    model.train()
    # model.eval()

    render_loss = []
    size_loss = []
    mask_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")

    # 在开始前清理一次GPU缓存
    torch.cuda.empty_cache()
    
    for i in pbar:
        if args.use_scenes and i % 10 == 0:
            args.scene_name = args.scene_list[torch.randint(0, len(args.scene_list), (1,)).item()]
            print(f"Scene: {args.scene_name}")
        
        # logger.logger.info(f"Iteration {i}: Selected frame_cur={frame_cur}, frame_next={frame_next}")
        
        # 减小切换频率
        if i % 10 == 0:
            # 随机选取一个 frame，范围从 args.frame_start (包含) 到 args.frame_end (不包含)
            # frame_cur = i % args.frame_end // args.gof_size * args.gof_size
            frame_cur = torch.randint(args.frame_start, args.frame_end - 1, (1,)).item()
            frame_cur = frame_cur // args.gof_size * args.gof_size if not args.random_init else frame_cur
            frame_next = torch.randint(frame_cur + 1, min(args.frame_end, frame_cur+args.gof_size), (1,)).item()
            # frame_next = frame_cur + 1 + i % args.gof_size
            print(f"Frame: {frame_cur} -> {frame_next}")
            train_frame_setup(args, frame_cur, frame_next)

            model.refresh_settings(args)

        loss_render, size, loss_mask, loss_motion = model()
        loss_size = size

        lambda_size = args.lambda_size if i > args.iterations // 3 else 0.0
        lambda_motion = 0.01 if i > args.iterations // 3 else 0.0
        # lambda_size = args.lambda_size
        loss = loss_render + loss_size * lambda_size + loss_mask * 0.01 + loss_motion *lambda_motion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()

        render_loss.append(loss_render.item())
        size_loss.append(loss_size.item())
        mask_loss.append(loss_mask.item())
        total_loss.append(loss.item())

        # 日志记录
        logger.log_iteration(i, {
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'loss_motion': loss_motion.item() if hasattr(loss_motion, 'item') else loss_motion,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        }, additional_info={'frame_cur': frame_cur, 'frame_next': frame_next})

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'loss_motion': loss_motion.item() if hasattr(loss_motion, 'item') else loss_motion,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        if args.checkpoint is not None and int(i) == args.checkpoint:
            model_path = os.path.join(args.model_path.split('frame')[0], f'checkpoint_{args.checkpoint}.pth')
            logger.logger.info(f"Saving checkpoint to {model_path}")
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.log_checkpoint(model_path)

    model.buffer = None
    args.model_path = args.model_path.split('frame')[0]
    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.logger.info(f"Saved final model to {model_path}")

    loss_curve_path = os.path.join(args.model_path, 'loss_curve.png')
    plot_loss_curve(render_loss, size_loss, total_loss, loss_curve_path)
    logger.logger.info(f"Saved loss curves to {loss_curve_path}")

    logger.log_training_complete()

def train_frame_jointly(args):
    # 创建Logger
    logger = args.logger
    
    model = FCGS_D(args).cuda()

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))
        logger.logger.info(f"Loaded checkpoint from {args.checkpoint_path}")

    model.train()

    render_loss = []
    size_loss = []
    mask_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")
    
    # 在开始前清理一次GPU缓存
    torch.cuda.empty_cache()
    time_ttl = 0
    for i in pbar:
        if args.use_scenes and i % 10 == 0:
            # 场景切换前先清理缓存
            torch.cuda.empty_cache()
            args.scene_name = args.scene_list[torch.randint(0, len(args.scene_list), (1,)).item()]
            print(f"Scene: {args.scene_name}")

        # 减小切换频率
        if i % 10 == 0:
            # 随机选取一个 frame，范围从 args.frame_start (包含) 到 args.frame_end (不包含)
            frame = torch.randint(args.frame_start, args.frame_end, (1,)).item()
            train_frame_setup(args, frame)
            
            # 此处是时间瓶颈！读写ply比较耗时！
            start_time = time.time()
            model.refresh_settings(args)
            time_ttl += time.time() - start_time


        loss_render, size, loss_mask = model()
        loss_size = size
        
        loss = loss_render + loss_size * args.lambda_size + loss_mask * 0.01
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()

        render_loss.append(loss_render.item())
        size_loss.append(loss_size.item())
        mask_loss.append(loss_mask.item())
        total_loss.append(loss.item())

        # 日志记录
        logger.log_iteration(i, {
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        }, additional_info={'frame': frame})

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        if args.checkpoint is not None and int(i) == args.checkpoint:
            model_path = os.path.join(args.model_path.split('frame')[0], f'checkpoint_{args.checkpoint}.pth')
            logger.logger.info(f"Saving checkpoint to {model_path}")
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.log_checkpoint(model_path)

    print(f"Total time: {time_ttl:.2f}s")

    args.model_path = args.model_path.split('frame')[0]
    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.logger.info(f"Saved final model to {model_path}")
    
    loss_curve_path = os.path.join(args.model_path, 'loss_curve.png')
    plot_loss_curve(render_loss, size_loss, total_loss, loss_curve_path)
    logger.logger.info(f"Saved loss curves to {loss_curve_path}")

    logger.log_training_complete()

def train(args):
    logger = args.logger
    
    logger.logger.info("Starting training...")

    model = FCGS_D(args).cuda()
    model.train()
    model.refresh_settings(args)

    render_loss = []
    size_loss = []
    mask_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")
    
    for i in pbar:
        loss_render, size, loss_mask = model()
        loss_size = size

        loss = loss_render + loss_size * args.lambda_size + loss_mask * 0.01
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.empty_cache()

        render_loss.append(loss_render.item())
        size_loss.append(loss_size.item())
        mask_loss.append(loss_mask.item())
        total_loss.append(loss.item())

        # 日志记录
        logger.log_iteration(i, {
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        if args.checkpoint is not None and int(i) == args.checkpoint:
            model_path = os.path.join(args.model_path, f'checkpoint_{args.checkpoint}.pth')
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            logger.log_checkpoint(model_path)

    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.logger.info(f"Saved final model to {model_path}")
    
    loss_curve_path = os.path.join(args.model_path, 'loss_curve.png')
    plot_loss_curve(render_loss, size_loss, total_loss, loss_curve_path)
    logger.logger.info(f"Saved loss curves to {loss_curve_path}")

    
    logger.log_training_complete()

def conduct_compress_decompress_as_gof(args, frame_start, frame_end):

    for frame in range(frame_start, frame_end):
        print(frame)
        train_frame_setup(args, frame-1, frame)
        os.makedirs(args.model_path, exist_ok=True)

        model_path = os.path.join(args.model_path.split('frame')[0], 'model.pth')
        if args.use_scenes:
            model_path = model_path.split(f'{args.scene_name}')[0] + '/model.pth'

        if frame % args.gof_size == 0:
            print('I frame')
        elif frame % args.gof_size == 1:
            conduct_compress(args, model_path, args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), args.next_3dgs)
            buffer_gaussian = conduct_decompress(args, model_path, args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), os.path.join(args.model_path, 'output.ply'))
        else:
            conduct_compress(args, model_path, args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), args.next_3dgs, use_buffer=True, buffer_gaussian=buffer_gaussian)
            conduct_decompress(args, model_path, args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), os.path.join(args.model_path, 'output.ply'))
            
def conduct_compress_at_frame(args, frame) :
    logger = args.logger
    
    if args.gof_size == 1:
        train_frame_setup(args, frame-1, frame)
    else:
        train_frame_setup(args, frame // args.gof_size * args.gof_size, frame)
    os.makedirs(args.model_path, exist_ok=True)
    logger.logger.info(f"Compressing frame {frame} at {args.model_path}")

    model_path = os.path.join(args.model_path.split('frame')[0], 'model.pth')
    if args.use_scenes:
        model_path = model_path.split(f'{args.scene_name}')[0] + '/model.pth'
    conduct_compress(args, model_path, args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), args.next_3dgs)

def conduct_decompress_at_frame(args, frame):
    logger = args.logger

    if args.gof_size == 1:
        train_frame_setup(args, frame-1, frame)
    else:
        train_frame_setup(args, frame // args.gof_size * args.gof_size, frame)
    os.makedirs(args.model_path, exist_ok=True)
    logger.logger.info(f"Decompressing frame {frame} at {args.model_path}")
    
    model_path = os.path.join(args.model_path.split('frame')[0], 'model.pth')
    if args.use_scenes:
        model_path = model_path.split(f'{args.scene_name}')[0] + '/model.pth'
    conduct_decompress(args, model_path, args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), os.path.join(args.model_path, 'output.ply'))

def conduct_compress(args, model_path, init_3dgs_path, ntc_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', nxt_gaussians_path=None, use_buffer=False, buffer_gaussian=None):
    logger = args.logger
    logger.logger.info(f"Compressing model from {init_3dgs_path}")
    
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        print(init_3dgs_path)
        print(nxt_gaussians_path)

        res = myFCGS_D.compress(init_3dgs_path, ntc_path, y_hat_bit_path, z_hat_bit_path, mask_path, dynamicGS_type=args.dynamicGS_type, nxt_gaussians_path=nxt_gaussians_path, use_buffer=use_buffer, buffer_gaussian=buffer_gaussian)
        with open(os.path.join(args.model_path, 'size.json'), 'w') as f:
            json.dump(res, f, indent=True)
        
        logger.logger.info(f'Compression Result: {res}') 

def conduct_decompress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', save_ply_path = './output.ply'):
    logger = args.logger
    logger.logger.info(f"Decompressing model from {init_3dgs_path}")
    
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene

        rec_gaussians = myFCGS_D.decompress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path, mask_path, dynamicGS_type=args.dynamicGS_type)
        # rec_gaussians.save_ply(save_ply_path)
        validate(rec_gaussians, scene, args,  args.model_path, save_img=True, logger=logger)
    return rec_gaussians
        
def validate(gaussians, scene, args, save_path='', save_img=False, logger=None):
        
    logger.logger.info("Starting validation...")
    
    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
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

            logger.logger.info(f"View {id}: PSNR={psnr_test[id]:.4f}, SSIM={ssim_test[id]:.4f}, LPIPS={lpips_test[id]:.4f}, L1={L1_test[id]:.4f}")

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

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FCGS_D')
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--Q_y', type=float, default=1e-3, help='granularity of quantization')
    parser.add_argument('--Q_z', type=float, default=1, help='granularity of quantization')
    parser.add_argument('--gof_size', type=int, default=20, help='number of gaussian frames in a group')
    parser.add_argument('--gaussian_feature_dim', type=int, default=56, help='dimension of gaussian feature')
    parser.add_argument('--motion_dim', type=int, default=7, help='dimension of motion')
    parser.add_argument('--hidden_dim', type=int, default=64, help='dimension of hidden')
    parser.add_argument('--lat_dim', type=int, default=16, help='dimension of latent')
    parser.add_argument('--init_3dgs', type=str, default=None, help='path to initial 3dgs')
    parser.add_argument('--next_3dgs', type=str, default=None, help='path to next 3dgs')
    parser.add_argument('--motion_estimator_path', type=str, default=None, help='path to motion estimator')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=300)
    parser.add_argument('--dynamicGS_type', type=str, default='3dgstream')
    parser.add_argument('--scene_name', type=str, default='cook_spinach')
    parser.add_argument('--scene_list', nargs='+', default=['sear_steak-5'])
    parser.add_argument('--use_first_as_test', action='store_true', help='use first frame as test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--conduct_training', action='store_true', help='conduct training')
    parser.add_argument('--conduct_compress', action='store_true', help='conduct compress')
    parser.add_argument('--conduct_decompress', action='store_true', help='conduct decompress')
    parser.add_argument('--scaler_y', type=int, default=1, help='scaler of y')
    parser.add_argument('--lambda_size', type=float, default=1e-3, help='scaler of z')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate')
    parser.add_argument('--per_frame', action='store_true', help='train per frame')
    parser.add_argument('--joint', action='store_true', help='train jointly')
    parser.add_argument('--test_frame_start', type=int, default=1)
    parser.add_argument('--test_frame_end', type=int, default=300)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--norm_radius', type=float, default=3)
    parser.add_argument('--residual_num', type=int, default=1e3)
    parser.add_argument('--use_scenes', action='store_true', help='use scenes')
    parser.add_argument('--logger', type=Logger, default=None)
    parser.add_argument('--dataset', type=str, default='dynerf')
    parser.add_argument('--use_gof', action='store_true', help='use group of frames')
    parser.add_argument('--random_init', action='store_true', help='randomly initialize the I frame')
    parser.add_argument('--motion_limit', type=float, default=0.1, help='motion limit')


    args = parser.parse_args()

    logger = Logger(args.model_path)
    logger.log_args(args)

    args.logger = logger

    torch.cuda.set_device(args.gpu)

    # model = FCGS_D(args).cuda()
    # model.eval()
    # for scene_name in args.scene_list:
    #     args.scene_name = scene_name
    #     print(f"Scene: {scene_name}")
    #     for frame in range(args.frame_start, args.frame_end):
    #         print(f"Frame: {frame}")
    #         train_frame_setup(args, frame)
    #         model.refresh_settings(args)
    #         model.test()


    if args.per_frame:
        train_frame(args)
    
    if args.joint:
        if args.conduct_training:
            if args.dynamicGS_type == '3dgstream':
                train_frame_jointly(args)
            elif args.dynamicGS_type == '3dgstream_explicit' or args.dynamicGS_type == '3dgstream_implicit':
                train_frame_jointly_stepping(args)
        
        if not args.use_scenes:
            args.scene_list = [args.scene_name]
        
        ori_model_path = args.model_path

        if args.conduct_compress and args.conduct_decompress:
            for scene_name in args.scene_list:
                args.scene_name = scene_name
                if args.use_scenes:
                    args.model_path = os.path.join(ori_model_path, scene_name)
                conduct_compress_decompress_as_gof(args, args.test_frame_start, args.test_frame_end)

        # if args.conduct_compress:
        #     for scene_name in args.scene_list:
        #         args.scene_name = scene_name
        #         if args.use_scenes:
        #             args.model_path = os.path.join(ori_model_path, scene_name)
        #         for frame in range(args.test_frame_start, args.test_frame_end):
        #             conduct_compress_at_frame(args, frame)
        #             torch.cuda.empty_cache()
        # if args.conduct_decompress:
        #     for scene_name in args.scene_list:
        #         args.scene_name = scene_name
        #         if args.use_scenes:
        #             args.model_path = os.path.join(ori_model_path, scene_name)
        #         for frame in range(args.test_frame_start, args.test_frame_end):
        #             conduct_decompress_at_frame(args, frame)

    if args.use_gof:
        if args.conduct_training:
            train_frame_gof(args)
        
        if not args.use_scenes:
            args.scene_list = [args.scene_name]
        ori_model_path = args.model_path

        
        if args.conduct_compress and args.conduct_decompress:
            for scene_name in args.scene_list:
                args.scene_name = scene_name
                if args.use_scenes:
                    args.model_path = os.path.join(ori_model_path, scene_name)
                conduct_compress_decompress_as_gof(args, args.test_frame_start, args.test_frame_end)
        
