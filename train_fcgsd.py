import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import os
import json
import torchvision
from scene import GaussianModel, Scene
from model.FCGS_D_model import FCGS_D

from gaussian_renderer import render
import lpips
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

def read_gaussian_file( file_path, sh_degree = 3):
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(file_path)
    return gaussians

def print_grad_fn(fn, depth=0):
    indent = "    " * depth
    print(f"{indent}{fn}")
    if hasattr(fn, "next_functions"):
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                print_grad_fn(next_fn, depth + 1)

def train_frame_setup(args, frame):
    if frame == args.frame_start:
        args.init_3dgs = f'/SDD_D/zwk/init_3dgs/{args.scene_name}/frame000000/point_cloud/iteration_7000/point_cloud.ply'
        args.motion_estimator_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.scene_name}/frame000001/NTC.pth'
    else:
        args.init_3dgs = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.scene_name}/frame{frame-1:06d}/point_cloud/iteration_150/point_cloud.ply'
        args.motion_estimator_path = f'/SSD2/chenzx/Projects/FCGS/output_gt/{args.scene_name}/frame{frame:06d}/NTC.pth'
    
    args.source_path = f'/SDD_D/zwk/data_dynamic/dynerf/{args.scene_name.split("-")[0]}/frame{frame:06d}'
    args.model_path = os.path.join(args.model_path.split('frame')[0], f'frame{frame:06d}')
        
def train_frame(args):
    for frame in range(args.frame_start, args.frame_end):
        train_frame_setup(args, frame)

        if args.conduct_training:
            print(f"\n[Training frame {frame}]")
            train(args)

        if args.conduct_compress:
            print(f"\n[Compressing frame {frame}]")
            conduct_compress(args, os.path.join(args.model_path, 'model.pth'), args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'))
        
        if args.conduct_decompress:
            print(f"\n[Decompressing frame {frame}]")
            conduct_decompress(args, os.path.join(args.model_path, 'model.pth'), args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), os.path.join(args.model_path, 'output.ply'))
        
        print('\n')

def train_frame_jointly(args):
    model = FCGS_D(args).cuda()

    if args.checkpoint_path is not None:
        model.load_state_dict(torch.load(args.checkpoint_path))

    model.train()

    render_loss = []
    size_loss = []
    mask_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")

    for i in pbar:
        # 随机选取一个 frame，范围从 args.frame_start (包含) 到 args.frame_end (不包含)
        frame = torch.randint(args.frame_start, args.frame_end, (1,)).item()
        train_frame_setup(args, frame)
        model.refresh_settings(args)

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

        pbar.set_postfix({
            'loss_render': loss_render.item() if hasattr(loss_render, 'item') else loss_render,
            'loss_size': loss_size.item() if hasattr(loss_size, 'item') else loss_size,
            'loss_mask': loss_mask.item() if hasattr(loss_mask, 'item') else loss_mask,
            'total_loss': loss.item() if hasattr(loss, 'item') else loss
        })

        if args.checkpoint is not None and int(i) == args.checkpoint:
            model_path = os.path.join(args.model_path.split('frame')[0], f'checkpoint_{args.checkpoint}.pth')
            print(model_path)
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        # # 打印 模型中 的 MotionEncoder 的参数本身
        # for name, param in model.named_parameters():
        #     if 'MotionEncoder' in name:
        #         print(name, param)


        
        # 计算梯度范数
        # total_norm = 0
        # for p in model.parameters():
        #     if p.grad is not None:
        #         param_norm = p.grad.data.norm(2)
        #         total_norm += param_norm.item() ** 2
        # total_norm = total_norm ** 0.5
        # print(f"Gradient norm: {total_norm}")

    args.model_path = args.model_path.split('frame')[0]
    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
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

    plt.savefig(os.path.join(args.model_path, 'loss_curve.png'))
    plt.show()
    plt.clf()  # 清除当前图形，防止后续绘图叠加在之前的图上

def train(args):

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
        # loss = loss_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.cuda.empty_cache()

        render_loss.append(loss_render.item())
        size_loss.append(loss_size.item())
        mask_loss.append(loss_mask.item())
        total_loss.append(loss.item())

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

    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
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

    plt.savefig(os.path.join(args.model_path, 'loss_curve.png'))
    plt.show()
    plt.clf()  # 清除当前图形，防止后续绘图叠加在之前的图上

def conduct_compress_at_frame(args, frame) :
    train_frame_setup(args, frame)
    os.makedirs(args.model_path, exist_ok=True)
    conduct_compress(args, os.path.join(args.model_path.split('frame')[0], 'model.pth'), args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'))

def conduct_decompress_at_frame(args, frame):
    train_frame_setup(args, frame)
    os.makedirs(args.model_path, exist_ok=True)
    conduct_decompress(args, os.path.join(args.model_path.split('frame')[0], 'model.pth'), args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'mask.b'), os.path.join(args.model_path, 'output.ply'))

def conduct_compress(args, model_path, init_3dgs_path, ntc_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b'):
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()

        res = myFCGS_D.compress(init_3dgs_path, ntc_path, y_hat_bit_path, z_hat_bit_path, mask_path)
        with open(os.path.join(args.model_path, 'size.json'), 'w') as f:
            json.dump(res, f, indent=True)
        print('Compression Result: ',res)

def conduct_decompress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_path = 'mask.b', save_ply_path = './output.ply'):
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene

        rec_gaussians = myFCGS_D.decompress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path, mask_path)
        # rec_gaussians.save_ply(save_ply_path)
        validate(rec_gaussians, scene, args.model_path, save_img=True)
        
def validate(gaussians, scene, save_path='', save_img = False):
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

            # render_loss = (1-ssim_loss) * 0.2 + L1_loss * 0.8
            # print(render_loss)

            torch.cuda.empty_cache()

        ssim_avg = sum(ssim_test.values()) / len(views)
        Ll1_avg = sum(L1_test.values()) / len(views)
        lpips_avg = sum(lpips_test.values()) / len(views)
        psnr_avg = sum(psnr_test.values()) / len(views)

        print(f"Evaluation results: psnr: {psnr_avg:.4f}, ssim: {ssim_avg:.4f}, lpips: {lpips_avg:.4f}, Ll1: {Ll1_avg:.4f}")

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
    parser.add_argument('--gof_size', type=int, default=10, help='number of gaussian frames in a group')
    parser.add_argument('--gaussian_feature_dim', type=int, default=56, help='dimension of gaussian feature')
    parser.add_argument('--motion_dim', type=int, default=7, help='dimension of motion')
    parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of hidden')
    parser.add_argument('--lat_dim', type=int, default=32, help='dimension of latent')
    parser.add_argument('--init_3dgs', type=str, default=None, help='path to initial 3dgs')
    parser.add_argument('--motion_estimator_path', type=str, default=None, help='path to motion estimator')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=300)
    parser.add_argument('--dynamicGS_type', type=str, default='3dgstream')
    parser.add_argument('--scene_name', type=str, default='cook_spinach-3')
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
    parser.add_argument('--checkpoint', type=int, default=None)


    args = parser.parse_args()
    # print(args)

    torch.cuda.set_device(args.gpu)
    # print(torch.cuda.current_device())

    # model = FCGS_D(args).cuda()
    # model.train()
    # print(gt_gaussians._xyz)
    # print(model.cur_gaussians._xyz)
    # print(model.MotionEstimation('3dgstream', model.cur_gaussians._xyz))

    # gt Gaussian of next frame
    # gt_gaussians = read_gaussian_file(f'/SSD2/chenzx/Projects/FCGS/output_gt/flame_steak-4/frame000010/point_cloud/iteration_150/point_cloud.ply')
    # args.source_path = '/SDD_D/zwk/data_dynamic/dynerf/flame_steak/frame000010'
    # args.motion_estimator_path = '/SSD2/chenzx/Projects/FCGS/output_gt/flame_steak-4/frame000001/NTC.pth'
    # validate(gt_gaussians, Scene(args))
    # validate(read_gaussian_file('./output.ply'), Scene(args))

    if args.per_frame:
        train_frame(args)
    
    if args.joint:
        if args.conduct_training:
            train_frame_jointly(args) 
        if args.conduct_compress:
            for frame in range(args.test_frame_start, args.test_frame_end):
                conduct_compress_at_frame(args, frame)
        if args.conduct_decompress:
            for frame in range(args.test_frame_start, args.test_frame_end):
                conduct_decompress_at_frame(args, frame)