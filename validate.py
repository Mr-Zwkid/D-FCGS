from gaussian_renderer import render, GaussianModel

import torch
import torchvision
from argparse import ArgumentParser
import sys
from scene import Scene
import os
import json
import lpips
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
import tqdm
from arguments import ModelParams, PipelineParams, OptimizationParams

def read_gaussian_file(file_path, sh_degree = 3):
    with torch.no_grad():
        gaussians = GaussianModel(sh_degree)
        gaussians.load_ply(file_path)
    return gaussians

def validate(gaussians, scene, args, save_path='', save_img=False, logger=None):
        
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

            print(f"SSIM: {ssim_loss:.4f}, L1: {L1_loss:.4f}, LPIPS: {lpips_test[id]:.4f}, PSNR: {psnr_test[id]:.4f}")

            torch.cuda.empty_cache()

        ssim_avg = sum(ssim_test.values()) / len(views)
        Ll1_avg = sum(L1_test.values()) / len(views)
        lpips_avg = sum(lpips_test.values()) / len(views)
        psnr_avg = sum(psnr_test.values()) / len(views)
    print('save_path', save_path)

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

    parser = ArgumentParser(description="dataset_param")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--ply_path", default="./bitstreams/tmp/point_cloud.ply", type=str)
    parser.add_argument("--save_path", default="./bitstreams/tmp/point_cloud.ply", type=str)
    # parser.add_argument("--source_path", default="./path/to/scene/", type=str)
    parser.add_argument("--use_first_as_test", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    torch.cuda.set_device(args.gpu)

    args.use_first_as_test = True
    args.eval = True

    # Read the Gaussian file
    gaussians = read_gaussian_file(args.ply_path)

    scene = Scene(args, shuffle=False)
    # Validate the scene
    validate(gaussians, scene, args, save_path=args.save_path)