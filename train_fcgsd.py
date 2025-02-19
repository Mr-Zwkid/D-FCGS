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

lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

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
        args.motion_estimator_path = f'/SDD_D/zwk/output/{args.scene_name}/frame000001/NTC.pth'
    else:
        args.init_3dgs = f'/SDD_D/zwk/output/{args.scene_name}/frame{frame-1:06d}/point_cloud/iteration_150/point_cloud.ply'
        args.motion_estimator_path = f'/SDD_D/zwk/output/{args.scene_name}/frame{frame:06d}/NTC.pth'
    
    args.source_path = f'/SDD_D/zwk/data_dynamic/dynerf/{args.scene_name.split("-")[0]}/frame{frame:06d}'
    args.model_path = f'./3DGStream-Res/dynerf/{args.scene_name.split("-")[0]}/frame{frame:06d}'
        
def train_frame(args):
    for frame in range(args.frame_start, args.frame_end):
        print(f"\n[Training frame {frame}]")
        train_frame_setup(args, frame)
        train(args)

        print(f"\n[Compressing frame {frame}]")
        conduct_compress(args, os.path.join(args.model_path, 'model.pth'), args.init_3dgs, args.motion_estimator_path, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'))
        
        print(f"\n[Decompressing frame {frame}]")
        conduct_decompress(args, os.path.join(args.model_path, 'model.pth'), args.init_3dgs, os.path.join(args.model_path, 'motion.b'), os.path.join(args.model_path, 'motion_prior.b'), os.path.join(args.model_path, 'output.ply'))

        print('\n')

def train(args):

    model = FCGS_D(args).cuda()
    model.train()

    render_loss = []
    size_loss = []
    mask_loss = []
    total_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pbar = tqdm.tqdm(range(args.iterations), desc="Training", unit="iteration")
    
    for i in pbar:
        loss_render, size, loss_mask = model()
        loss_size = size / 256

        # loss = loss_render + loss_size * 0.0001 + loss_mask * 0.001
        loss = loss_render + loss_size * 0.001
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

    model_path = os.path.join(args.model_path, 'model.pth')
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    plt.plot(render_loss, label='render_loss')
    plt.plot(size_loss, label='size_loss')
    plt.plot(mask_loss, label='mask_loss')
    plt.plot(total_loss, label='total_loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.show()
    plt.savefig(os.path.join(args.model_path, 'loss_curve.png'))
    
def conduct_compress(args, model_path, init_3dgs_path, ntc_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b'):
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()

        res = myFCGS_D.compress(init_3dgs_path, ntc_path, y_hat_bit_path, z_hat_bit_path)
        with open(os.path.join(args.model_path, 'size.json'), 'w') as f:
            json.dump(res, f, indent=True)
        print('Compression Result: ',res)

def conduct_decompress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', save_ply_path = './output.ply'):
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene

        rec_gaussians = myFCGS_D.decompress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path)
        # rec_gaussians.save_ply(save_ply_path)
        validate(rec_gaussians, scene, args.model_path, save_img=True)
        
def validate(gaussians, scene, save_path='', save_img = False):
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

    parser.add_argument('--Q_y', type=int, default=0.1, help='granularity of quantization')
    parser.add_argument('--Q_z', type=int, default=1, help='granularity of quantization')
    parser.add_argument('--gof_size', type=int, default=10, help='number of gaussian frames in a group')
    parser.add_argument('--gaussian_feature_dim', type=int, default=56, help='dimension of gaussian feature')
    parser.add_argument('--motion_dim', type=int, default=7, help='dimension of motion')
    parser.add_argument('--hidden_dim', type=int, default=256, help='dimension of hidden')
    parser.add_argument('--lat_dim', type=int, default=64, help='dimension of latent')
    parser.add_argument('--init_3dgs', type=str, default=None, help='path to initial 3dgs')
    parser.add_argument('--motion_estimator_path', type=str, default=None, help='path to motion estimator')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=300)
    parser.add_argument('--dynamicGS_type', type=str, default='3dgstream')
    parser.add_argument('--scene_name', type=str, default='cook_spinach-3')

    args = parser.parse_args()
    print(args)

    # model = FCGS_D(args).cuda()
    # model.train()
    # print(gt_gaussians._xyz)
    # print(model.cur_gaussians._xyz)
    # print(model.MotionEstimation('3dgstream', model.cur_gaussians._xyz))

    # gt Gaussian of next frame
    # gt_gaussians = read_gaussian_file(f'/SDD_D/zwk/output/{scene_name}/frame000001/point_cloud/iteration_150/point_cloud.ply')
    # validate(gt_gaussians, Scene(args))
    # validate(read_gaussian_file('./output.ply'), Scene(args))

    train_frame(args)