import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import os
import json
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

        
    torch.save(model.state_dict(), 'model.pth')
    
    plt.plot(render_loss, label='render_loss')
    plt.plot(size_loss, label='size_loss')
    plt.plot(mask_loss, label='mask_loss')
    plt.plot(total_loss, label='total_loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.show()
    plt.savefig('loss_curve.png')
    
def conduct_compress(args, model_path, init_3dgs_path, ntc_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b'):
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()

        res = myFCGS_D.compress(init_3dgs_path, ntc_path,y_hat_bit_path, z_hat_bit_path)
        print(res)

def conduct_decompress(args, model_path, init_3dgs_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', save_ply_path = './output.ply'):
    with torch.no_grad():
        model = torch.load(model_path)
        myFCGS_D = FCGS_D(args)
        myFCGS_D.load_state_dict(model, strict=False)
        myFCGS_D.eval()
        myFCGS_D.cuda()
        scene = myFCGS_D.scene

        rec_gaussians = myFCGS_D.decompress(init_3dgs_path, y_hat_bit_path, z_hat_bit_path)
        rec_gaussians.save_ply(save_ply_path)
        validate(rec_gaussians, scene)
        
def validate(gaussians, scene):
    views = scene.getTestCameras()
    # views = scene.getTrainCameras()
    with torch.no_grad():
        ssim_test_sum = 0
        L1_test_sum = 0
        lpips_test_sum = 0
        psnr_test_sum = 0
        curr_rendering_list = []
        for _, view in enumerate(tqdm.tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipe=args, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))[
                "render"]  # [3, H, W]
            gt = view.original_image[0:3, :, :].to("cuda")
            rendering = torch.round(rendering.mul(255).clamp_(0, 255)) / 255.0
            ssim_test_sum += (ssim(rendering, gt)).mean().double().item()
            L1_test_sum += l1_loss(rendering, gt).mean().double().item()
            lpips_test_sum += lpips_fn(rendering, gt).mean().double().item()
            psnr_test_sum += psnr(rendering, gt).mean().double().item()
            curr_rendering_list.append(rendering)

            ssim_loss = ssim(rendering, gt).item()
            L1_loss = l1_loss(rendering, gt).item()
            render_loss = (1-ssim_loss) * 0.2 + L1_loss * 0.8
            print(render_loss)

            torch.cuda.empty_cache()
        ssim_avg = ssim_test_sum / len(views)
        Ll1_avg = L1_test_sum / len(views)
        lpips_avg = lpips_test_sum / len(views)
        psnr_avg = psnr_test_sum / len(views)

        print(f"Evaluation results: psnr: {psnr_avg:.4f}, ssim: {ssim_avg:.4f}, lpips: {lpips_avg:.4f}, Ll1: {Ll1_avg:.4f}")

    with open(os.path.join('psnr.json'), 'w') as f:
        data = {'PSNR': psnr_avg}
        json.dump(data, f, indent=True)
    
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
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--dynamicGS_type', type=str, default='3dgstream')
    # parser.add_argument('--iterations', type=int, default=1000)

    args = parser.parse_args()

    # init_3dgs = '/SDD_D/zwk/init_3dgs/sear_steak-5/frame000000/point_cloud/iteration_7000/point_cloud.ply'
    init_3dgs = '/SDD_D/zwk/output/sear_steak-5/frame000003/point_cloud/iteration_150/point_cloud.ply'
    ntc = '/SDD_D/zwk/output/sear_steak-5/frame000004/NTC.pth'

    gt_gaussians = read_gaussian_file('/SDD_D/zwk/output/sear_steak-5/frame000004/point_cloud/iteration_150/point_cloud.ply')

    args.init_3dgs = init_3dgs
    args.motion_estimator_path = ntc
    args.source_path = '/SDD_D/zwk/data_dynamic/dynerf/sear_steak/frame000004'
    args.model_path = './outputs/dynerf/sear_steak'
    args.images = 'images_2'
    args.eval = True
    args.iterations = 1
    # print(args)

    # model = FCGS_D(args).cuda()
    # model.train()

    # print(gt_gaussians._xyz)
    # print(model.cur_gaussians._xyz)
    # print(model.MotionEstimation('3dgstream', model.cur_gaussians._xyz))

    # gt Gaussian of next frame
    # validate(gt_gaussians, Scene(args))
    # validate(read_gaussian_file('./output.ply'), Scene(args))
    train(args)
    conduct_compress(args, './model.pth', init_3dgs, ntc)
    conduct_decompress(args, './model.pth', init_3dgs)
