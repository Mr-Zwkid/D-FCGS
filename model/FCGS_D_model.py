import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import copy
from scene import GaussianModel, Scene, SimpleGaussianModel
from model.gpcc_utils import sorted_voxels, sorted_orig_voxels, compress_gaussian_params, decompress_gaussian_params
from model.encodings_cuda import STE_multistep, encoder_gaussian, decoder_gaussian, encoder_gaussian_chunk, decoder_gaussian_chunk, \
    encoder_gaussian_mixed, decoder_gaussian_mixed, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk,\
    encoder_factorized, decoder_factorized, encoder_factorized_chunk, decoder_factorized_chunk, encoder, decoder
from model.entropy_models import Entropy_factorized, Entropy_gaussian
from model.subnet import GDN1D, MaskedConv1d, DownsampleLayer, ClipLayer
from random import randint
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from model.motion_estimators import NeuralTransformationCache
from model.grid_utils import normalize_xyz, _grid_creater, _grid_encoder, FreqEncoder
from pytorch3d.ops import sample_farthest_points


bit_to_MB = 8 * 1024 * 1024

def plot_hist(x, title='histogram', bins=100, range=None, save_path=None):
    import matplotlib.pyplot as plt
    x = x.view(-1).detach().cpu().numpy()
    plt.hist(x, bins=bins, range=range)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def GaussianParameterPack(gaussians):
    g_xyz = gaussians._xyz.detach() # [N, 3]
    N_gaussian = g_xyz.shape[0] # N
    _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
    _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
    _opacity = gaussians._opacity.detach()  # [N, 1]
    _scaling = gaussians._scaling.detach()  # [N, 3]
    _rotation = gaussians._rotation.detach()  # [N, 4]

    _opacity[torch.isinf(_opacity)] = _opacity[~torch.isinf(_opacity)].max()

    # print('g_xyz: ', g_xyz.max(), g_xyz.min())
    # print('features_dc: ', _features_dc.max(), _features_dc.min())
    # print('features_rest: ', _features_rest.max(), _features_rest.min())
    # print('opacity: ', _opacity.max(), _opacity.min())
    # print('scaling: ', _scaling.max(), _scaling.min())
    # print('rotation: ', _rotation.max(), _rotation.min())

    g_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation], dim=-1)  # [N, 56]
    return g_xyz, g_fea, N_gaussian

def GaussianParameterUnpack(g_fea, N_gaussian):
    _features_dc = g_fea[:, :3].view(N_gaussian, 1, 3)
    _features_rest = g_fea[:, 3:48].view(N_gaussian, 15, 3)
    _opacity = g_fea[:, 48].view(N_gaussian, 1)
    _scaling = g_fea[:, 49:52]
    _rotation = g_fea[:, 52:]
    return _features_dc, _features_rest, _opacity, _scaling, _rotation

def quaternion_multiply(a, b):
    """
    Multiply two sets of quaternions.
    
    Parameters:
    a (Tensor): A tensor containing N quaternions, shape = [N, 4]
    b (Tensor): A tensor containing N quaternions, shape = [N, 4]
    
    Returns:
    Tensor: A tensor containing the product of the input quaternions, shape = [N, 4]
    """
    a_norm=torch.nn.functional.normalize(a)
    b_norm=torch.nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)

class FCGS_D(nn.Module):
    def __init__(self, args):
        super(FCGS_D, self).__init__()
        self.args = args
        self.Q_y = args.Q_y
        self.Q_z = args.Q_z
        self.scaler_y = args.scaler_y
        self.gof_size = args.gof_size
        self.gaussian_position_dim = 3
        self.gaussian_feature_dim = args.gaussian_feature_dim # 56
        self.motion_dim = args.motion_dim # 7
        self.hidden_dim = args.hidden_dim # 256
        self.lat_dim = args.lat_dim 
        self.GDN = GDN1D
        self.norm_radius = args.norm_radius
        self.residual_num = args.residual_num
        self.motion_limit = args.motion_limit

        self.cur_gaussians = None 
        self.nxt_gaussians = None

        self.scene = None

        self.viewpoint_stack = None

        self.MotionEstimator = None

        if args.init_3dgs:
            self.refresh_settings(args)

        self.GridEncoder_config = {
                "encoding_hash": {
                            "otype": "HashGrid",
                            "n_dims_to_encode": 3,
                            "per_level_scale": 2.0,
                            "log2_hashmap_size": 16,
                            "base_resolution": 16,
                            "n_levels": 16,
                            "n_features_per_level": 8
                },

                "encoding_freq": {
                    "otype": "Frequency",
                    "n_frequencies": 12  
                },

                "network": {
                    "otype": "FullyFusedMLP",
                    "activation": "LeakyReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 3
                },
            }
        
        self.ResidualGenerator = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.gaussian_feature_dim)
        )

        self.GridEncoder = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_hash'], network_config=self.GridEncoder_config['network'])

        self.FreqEncoder = tcnn.NetworkWithInputEncoding(n_input_dims=self.motion_dim, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_freq'], network_config=self.GridEncoder_config['network'])
        
        self.MotionWeighting = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.motion_dim, encoding_config=self.GridEncoder_config['encoding_freq'], network_config=self.GridEncoder_config['network'])

        self.GaussianFeatureExtractor = nn.Sequential(
            nn.Linear(self.gaussian_feature_dim, self.hidden_dim),
            # self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            # self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim)
        )

        self.MotionMaskGenerator = nn.Sequential(
            # nn.Linear(self.lat_dim, self.hidden_dim),
            nn.Linear(self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        # self.FramePositionExtractor = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_hash'], network_config=self.GridEncoder_config['network'])
        self.FramePositionExtractor = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_freq'], network_config=self.GridEncoder_config['network'])
        self.FrameRotationExtractor = tcnn.NetworkWithInputEncoding(n_input_dims=4, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_freq'], network_config=self.GridEncoder_config['network'])
        # self.FrameRotationExtractor = DownsampleLayer(4, self.hidden_dim, self.lat_dim, 0)
        self.Combiner = nn.Sequential(
            nn.Linear(2 * self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.motion_dim)
        )

        # self.MotionEncoder = DownsampleLayer(args.motion_dim, args.hidden_dim, args.lat_dim, 3)
        # self.MotionDecoder = DownsampleLayer(args.lat_dim, args.hidden_dim, args.motion_dim, 3)

        # self.MotionEncoder = nn.Sequential(
        #     nn.Conv1d(args.motion_dim , args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.lat_dim, 1),
        #     # nn.BatchNorm1d(args.lat_dim),
        # )

        # self.MotionDecoder = nn.Sequential(
        #     nn.Conv1d(args.lat_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.motion_dim, 1),
        #     # nn.Tanh(),
        #     # ClipLayer(-0.3, 0.3)
        # )

        self.MotionEncoder = nn.Sequential(
            nn.Linear(self.motion_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim),
        )

        self.MotionDecoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.motion_dim),
            # nn.Tanh(),
            ClipLayer(-self.motion_limit, self.motion_limit)
        )

        # self.MotionPriorEncoder = nn.Sequential(
        #     nn.Conv1d(args.lat_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.lat_dim, 1)
        # )

        # self.MotionPriorDecoder = nn.Sequential(
        #     nn.Conv1d(args.lat_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.hidden_dim, 1),
        #     nn.BatchNorm1d(args.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(args.hidden_dim, args.lat_dim, 1)
        # )

        self.MotionPriorEncoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim),
        )

        self.MotionPriorDecoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim),
        )

        self.AutoRegressiveMotion = nn.Sequential(
            MaskedConv1d(in_channels=args.lat_dim, out_channels=args.hidden_dim, kernel_size=3, stride=1, padding=1, mask_type="A"),
            nn.LeakyReLU(inplace=True),
            MaskedConv1d(in_channels=args.hidden_dim, out_channels=args.hidden_dim, kernel_size=3, stride=1, padding=1, mask_type="A"),
            nn.LeakyReLU(inplace=True),
            MaskedConv1d(in_channels=args.hidden_dim, out_channels=args.lat_dim, kernel_size=3, stride=1, padding=1, mask_type="A"),
            nn.LeakyReLU(inplace=True)
        )

        # self.EntropyParametersMotion = nn.Sequential (
        #     nn.Conv1d(2 * self.lat_dim, self.hidden_dim, 1),
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(self.hidden_dim, self.hidden_dim, 1),
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv1d(self.hidden_dim, 2 * self.lat_dim, 1)
        # )

        
        self.EntropyParametersMotion = nn.Sequential(
            nn.Linear(2 * self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2 * self.lat_dim)
        )

        self.EntropyFactorizedMotion = Entropy_factorized(self.lat_dim, Q=self.Q_z)

        self.EntropyGaussianMotion = Entropy_gaussian(Q=self.Q_y)

        self.AdaptiveQuantizationY = nn.Sequential(
            nn.Conv1d(self.lat_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, 1, 1),
            nn.Tanh()
        )

        self.AdaptiveQuantizationZ = nn.Sequential(
            nn.Conv1d(self.lat_dim, self.hidden_dim, 1),
            nn.BatchNorm1d(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(self.hidden_dim, 1, 1),
            nn.Sigmoid()
        )
   
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # 获取完整 state_dict
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # 删除不需要保存的项，这里假设所有以 "MotionEstimator"、“cur_gaussians” 或 “nxt_gaussians” 开头的键删除
        keys_to_remove = [key for key in state.keys() if key.startswith('MotionEstimator') 
                                              or key.startswith('cur_gaussians') 
                                              or key.startswith('nxt_gaussians')
                                              or key.startswith('scene')
                                              or key.startswith('viewpoint_stack')
                                              or key.startswith('buffer')]
        
        for key in keys_to_remove:
            del state[key]
        return state
    
    def quantize(self, x, Q=1, train_flag=True):
        if not train_flag:
            x_q = STE_multistep.apply(x, Q)
        else:
            # add uniform noise to simulate quantization while training
            x_q = x + torch.empty_like(x).uniform_(-0.5, 0.5) * Q  
        return self.clamp(x_q, Q)
    
    def clamp(self, x, Q):
        x[torch.isinf(x)] = 0
        x[torch.isnan(x)] = 0 
        x_mean = x.mean().detach()
        x_min = x_mean - 15_000 * Q
        x_max = x_mean + 15_000 * Q
        x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        return x

    def get_xyz_bound(self, gaussians, percentile=86.6):
        half_percentile = (100 - percentile) / 200
        xyz_bound_min = torch.quantile(gaussians._xyz,half_percentile,dim=0)
        xyz_bound_max = torch.quantile(gaussians._xyz,1 - half_percentile,dim=0)
        return xyz_bound_min, xyz_bound_max

    def ComputeRenderLoss(self, gaussians, training=True):

        bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy() if training else self.scene.getTestCameras().copy()
        viewpoint_camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_camera, gaussians, self.args, background)
        rendered_img = render_pkg["render"]
        # TODO use depth or optical flow as prior

        gt_img = viewpoint_camera.original_image.cuda()

        L1_loss = l1_loss(rendered_img, gt_img)
        ssim_loss = ssim(rendered_img, gt_img)
        render_loss = (1.0 - self.args.lambda_dssim) * L1_loss + self.args.lambda_dssim * (1.0 - ssim_loss)
        
        # train
        viewpoint_stack = self.scene.getTrainCameras().copy()
        len_viewpoint_stack = len(viewpoint_stack)
        cur_psnr = 0
        while len(viewpoint_stack) > 0:
            viewpoint_camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            render_pkg = render(viewpoint_camera, gaussians, self.args, background)
            rendered_img = render_pkg["render"]
            gt_img = viewpoint_camera.original_image.cuda()
            cur_psnr += psnr(rendered_img, gt_img).mean().double().item()
        print(f"PSNR_train: {cur_psnr / len_viewpoint_stack:.2f} dB")

        # test
        viewpoint_stack = self.scene.getTestCameras().copy()
        len_viewpoint_stack = len(viewpoint_stack)
        cur_psnr = 0
        while len(viewpoint_stack) > 0:
            viewpoint_camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            render_pkg = render(viewpoint_camera, gaussians, self.args, background)
            rendered_img = render_pkg["render"]
            gt_img = viewpoint_camera.original_image.cuda()
            cur_psnr += psnr(rendered_img, gt_img).mean().double().item()
        print(f"PSNR_test: {cur_psnr / len_viewpoint_stack:.2f} dB")



        return render_loss

    def MotionEstimatorSetup(self, dynamicGS_type, motion_estimator_path=None):
        if dynamicGS_type == '3dgstream':
            self.MotionEstimator = NeuralTransformationCache(*self.get_xyz_bound(self.cur_gaussians))
            self.MotionEstimator.load_state_dict(torch.load(motion_estimator_path), strict=True)
            self.MotionEstimator.requires_grad_(False)
        elif dynamicGS_type == '3dgstream_explicit':
            pass
        elif dynamicGS_type == '3dgstream_implicit':
            pass
        # elif dynamicGS_type == 'deformable-gs':
        #     self.MotionEstimator = NeuralTransformationCache()
        # elif dynamicGS_type == '4d-gs':
        #     self.MotionEstimator = NeuralTransformationCache()
        else:
            raise ValueError('Invalid dynamicGS_type')
        
    def MotionEstimation(self, dynamicGS_type, cur_gaussians, nxt_gaussians=None):
        if dynamicGS_type == '3dgstream':
            xyz = cur_gaussians._xyz
            mask, d_xyz, d_rot = self.MotionEstimator(xyz)
            return torch.cat((d_xyz, d_rot), dim=1)
        elif dynamicGS_type == '3dgstream_explicit':
            cur_xyz = cur_gaussians._xyz
            nxt_xyz = nxt_gaussians._xyz

            cur_rot = cur_gaussians._rotation
            nxt_rot = nxt_gaussians._rotation
            assert cur_xyz.shape[0] == nxt_xyz.shape[0], 'The number of points in the current and next frame should be the same'
            
            # NOTE Avoid the rotation flip
            if abs(cur_rot.mean() + nxt_rot.mean()) < abs(cur_rot.mean()):
                nxt_rot = -nxt_rot
            
            d_xyz = nxt_xyz - cur_xyz
            d_rot = nxt_rot - cur_rot

            # print('d_xyz: ', d_xyz)
            
            return torch.cat((d_xyz, d_rot), dim=1)    
        elif dynamicGS_type == '3dgstream_implicit':
            cur_xyz = cur_gaussians._xyz.detach()
            nxt_xyz = nxt_gaussians._xyz.detach()


            # Min-max normalization for position
            cur_xyz = (cur_xyz - cur_xyz.min(dim=0)[0]) / (cur_xyz.max(dim=0)[0] - cur_xyz.min(dim=0)[0] + 1e-6)
            nxt_xyz = (nxt_xyz - nxt_xyz.min(dim=0)[0]) / (nxt_xyz.max(dim=0)[0] - nxt_xyz.min(dim=0)[0] + 1e-6)

            # Or alternatively, you could use mean normalization
            # cur_xyz = (cur_xyz - cur_xyz.mean(dim=0)) / (cur_xyz.std(dim=0) + 1e-6)
            # nxt_xyz = (nxt_xyz - nxt_xyz.mean(dim=0)) / (nxt_xyz.std(dim=0) + 1e-6)
    
            
            print('cur_xyz: ', cur_xyz.shape)
            print('nxt_xyz: ', nxt_xyz.shape)

            cur_xyz_trans = self.FramePositionExtractor(cur_xyz)
            nxt_xyz_trans = self.FramePositionExtractor(nxt_xyz)
            # print('cur_xyz_trans:', cur_xyz_trans)
            # print('nxt_xyz_trans:', nxt_xyz_trans)
            # print('difference: ', nxt_xyz_trans - cur_xyz_trans)

            cur_rot = cur_gaussians._rotation.detach()
            nxt_rot = nxt_gaussians._rotation.detach()
            # NOTE Avoid the rotation flip
            if abs(cur_rot.mean() + nxt_rot.mean()) < abs(cur_rot.mean()):
                nxt_rot = -nxt_rot

            cur_rot = (cur_rot - cur_rot.mean(dim=0)[0]) / (cur_rot.max(dim=0)[0] - cur_rot.min(dim=0)[0] + 1e-6)
            nxt_rot = (nxt_rot - nxt_rot.mean(dim=0)[0]) / (nxt_rot.max(dim=0)[0] - nxt_rot.min(dim=0)[0] + 1e-6)
            
            # print('cur_rot: ', cur_rot)
            # print('nxt_rot: ', nxt_rot)

            cur_rot_trans = self.FrameRotationExtractor(cur_rot)
            nxt_rot_trans = self.FrameRotationExtractor(nxt_rot)
            
            # print('cur_rot_trans', cur_rot_trans)
            # print('nxt_rot_trans', nxt_rot_trans)

            motion_xyz = (nxt_xyz_trans - cur_xyz_trans)

            motion_rot = (nxt_rot_trans - cur_rot_trans)

            motion = self.Combiner(torch.cat([motion_xyz, motion_rot], dim=1).to(torch.float32))
            # print(motion)

            return motion
     
    def MotionCompensation(self, dynamicGS_type, dec_motion, cur_gaussians, nxt_gaussians_=None):
        nxt_gaussians = SimpleGaussianModel(cur_gaussians)
        if dynamicGS_type == '3dgstream':
            # Ensure that the original _xyz is detached, so only dec_motion contributes gradients
            nxt_gaussians._xyz = cur_gaussians._xyz.detach() + dec_motion[:, :3]
            nxt_gaussians._rotation = quaternion_multiply(cur_gaussians._rotation.detach(), dec_motion[:, 3:])
        elif dynamicGS_type == '3dgstream_explicit':
            nxt_gaussians._xyz = cur_gaussians._xyz.detach() + dec_motion[:, :3]
            nxt_gaussians._rotation = cur_gaussians._rotation.detach() + dec_motion[:, 3:]
        elif dynamicGS_type == '3dgstream_implicit':
            nxt_gaussians._xyz = cur_gaussians._xyz.detach() + dec_motion[:, :3]
            nxt_gaussians._rotation = cur_gaussians._rotation.detach() + dec_motion[:, 3:]
            # nxt_gaussians.rotation = dec_motion[:, 3:]

        return nxt_gaussians

    def ResidualCompensation(self, cur_gaussians, residual_position=None, residual_feature=None): 
        residual_color_dc, residual_color_rest, residual_opacity, residual_scaling, residual_rotation = torch.split(residual_feature, [3, 45, 1, 3, 4], dim=1)

        if residual_position is not None:
            residual_position = residual_position.view(-1, 3)
            cur_gaussians._xyz = cur_gaussians._xyz + residual_position
        if residual_feature is not None:
            residual_color_dc = residual_color_dc.view(-1, 3)
            residual_color_rest = residual_color_rest.view(-1, 15, 3)
            residual_opacity = residual_opacity.view(-1, 1)
            residual_scaling = residual_scaling.view(-1, 3)
            residual_rotation = residual_rotation.view(-1, 4)
            # cur_gaussians._features_dc = cur_gaussians._features_dc + residual_color_dc
            # print(residual_color_rest)
            cur_gaussians._features_rest = cur_gaussians._features_rest + residual_color_rest
            # cur_gaussians._opacity = cur_gaussians._opacity + residual_opacity
            # cur_gaussians._scaling = cur_gaussians._scaling + residual_scaling
            # cur_gaussians._rotation = quaternion_multiply(cur_gaussians._rotation, residual_rotation)
            pass
        return cur_gaussians

    def refresh_settings(self, args):
        self.cur_gaussians = self.read_gaussian_file(args.init_3dgs, sh_degree=3)
        # print('cur_gaussians: ', args.init_3dgs)
        if args.next_3dgs is not None:
            # print('next_3dgs: ', args.next_3dgs)
            self.nxt_gaussians = self.read_gaussian_file(args.next_3dgs, sh_degree=3)
            # print('next_3dgs: ', self.nxt_gaussians._rotation)
        self.scene = Scene(args) 
        self.viewpoint_stack = None
        self.MotionEstimatorSetup(args.dynamicGS_type, args.motion_estimator_path)
    
    def buffer_loading(self, args, cur_gaussians, nxt_gaussians=None):
        self.cur_gaussians = cur_gaussians
        self.nxt_gaussians = self.read_gaussian_file(args.next_3dgs, sh_degree=3)
        self.scene = Scene(args) 
        self.viewpoint_stack = None
        self.MotionEstimatorSetup(args.dynamicGS_type, args.motion_estimator_path)

    def buffer_capture(self):
        return self.buffer

    def read_gaussian_file(self, file_path, sh_degree = 3):
        with torch.no_grad():
            gaussians = GaussianModel(sh_degree)
            gaussians.load_ply(file_path)
        return gaussians
    
    def init_test_gaussians(self, sh_degree = 3):
        self.cur_gaussians = self.read_gaussian_file('/SDD_D/zwk/output/cook_spinach-3-ori/init_3dgs.ply')
        self.nxt_gaussians = self.read_gaussian_file('/SDD_D/zwk/output/cook_spinach-3-ori/init_3dgs.ply')

    def compress(self, cur_gaussians_path, motion_estimator_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_bit_path = 'mask.b', dynamicGS_type='3dgstream', nxt_gaussians_path=None, use_buffer = False, buffer_gaussian = None):
        
        cur_gaussians = self.read_gaussian_file(cur_gaussians_path) if not use_buffer else buffer_gaussian
        nxt_gaussians = self.read_gaussian_file(nxt_gaussians_path) if nxt_gaussians_path is not None else None
        self.MotionEstimatorSetup(dynamicGS_type, motion_estimator_path)

        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(cur_gaussians)
        # cur_xyz, cur_fea = sorted_orig_voxels(cur_xyz, cur_fea)
        position_features = self.GridEncoder(cur_xyz)

        # normalize the position_features
        position_features = F.normalize(position_features, p=2, dim=1)

        extracted_features = self.GaussianFeatureExtractor(cur_fea) # N, latent_dim
        extracted_features = position_features + extracted_features

        ctx_params_motion = extracted_features
        # ctx_params_motion = extracted_features.view(1, self.lat_dim, -1) # [1, latent_dim, N]

        # ctx_params_motion = self.AutoRegressiveMotion(extracted_features) # [1, latent_dim, N]

        est_motion = self.MotionEstimation(dynamicGS_type, cur_gaussians, nxt_gaussians) # I-NGP in 3DGStream / MLP in Deformable-GS / Hexplane in 4D-GS

        # mask_motion_tmp = self.MotionMaskGenerator(est_motion.detach().to(torch.float32)) # [N, 1]
        # mask_motion_ = ((mask_motion_tmp > 0.01).float() - mask_motion_tmp).detach() + mask_motion_tmp  # [N, 1]
        # mask_motion = mask_motion_.to(torch.bool).squeeze()
        # est_motion = est_motion[mask_motion] # [N_m, motion_dim]

        # est_motion = est_motion.unsqueeze(0).permute(0, 2, 1).contiguous() # [1, motion_dim, N_m]
        
        y_motion = self.MotionEncoder(est_motion) # [1, latent_dim, N_m]

        # Q_y = (self.AdaptiveQuantizationY(ctx_params_motion) + 1) * self.Q_y # [N_m, latent_dim]
        Q_y = self.Q_y
        y_hat_motion = self.quantize(y_motion, Q=Q_y, train_flag=False) # [1, latent_dim, N_m]

        z_motion = self.MotionPriorEncoder(y_motion) 
        # Q_z = self.AdaptiveQuantizationZ(ctx_params_motion) * 2 * self.Q_z # [N_m, latent_dim]
        Q_z = self.Q_z
        z_hat_motion = self.quantize(z_motion, Q=Q_z, train_flag=False) # [1, latent_dim, N_m]
        params_motion = self.MotionPriorDecoder(z_hat_motion) # [1, latent_dim, N_m]

        distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1)) 

        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6
        mean_motion = mean_motion.contiguous()

        # y_hat_motion = y_hat_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, latent_dim]
        # z_hat_motion = z_hat_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, latent_dim]
        # Q_y = Q_y.squeeze(0).permute(1, 0).contiguous().repeat(1, self.lat_dim) # [N_m, latent_dim]

        plot_hist(est_motion, save_path=y_hat_bit_path.replace('.b', '_est_motion.png'), title='est_motion', range=(-0.3, 0.3), bins=100)
        print('est_motion: ', est_motion)
        print('est_motion: ', est_motion.max(), est_motion.min())
        # print('mean_motion: ', mean_motion)
        # print('std_motion: ', std_motion)  
        print('y_motion: ', y_motion)  
        print('y_hat_motion: ', y_hat_motion)
        print(y_hat_motion.sum())
        print('z_hat_motion: ', z_hat_motion)

        dec_motion = self.MotionDecoder(y_hat_motion) # [1, motion_dim, N_m]
        print('dec_motion: ', dec_motion, dec_motion.max(), dec_motion.min())
        print(dec_motion.sum(dim=1))

        # find the number of those y_hat_motion that are not zero in dim 0


        bits_mask = 0
        # bits_mask = encoder(mask_motion_, mask_bit_path)
        bits_motion = encoder_gaussian_chunk(y_hat_motion, mean_motion.contiguous(), std_motion, Q_y, y_hat_bit_path, chunk_size=100000)
        bits_prior_motion = encoder_factorized_chunk(z_hat_motion, self.EntropyFactorizedMotion._logits_cumulative, Q_z, z_hat_bit_path, chunk_size=10000)

        return {
            'bits_mask': bits_mask / bit_to_MB,
            'bits_motion': bits_motion / bit_to_MB,
            'bits_prior_motion': bits_prior_motion / bit_to_MB,
        }

    def decompress(self, cur_gaussians_path, y_hat_bit_path, z_hat_bit_path, mask_bit_path, dynamicGS_type='3dgstream'):
        cur_gaussians = self.read_gaussian_file(cur_gaussians_path)

        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(self.cur_gaussians)

        # norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(cur_xyz, K=self.norm_radius)
        position_features = self.GridEncoder(cur_xyz)

        # normalize the position_features
        position_features = F.normalize(position_features, p=2, dim=1)

        norm_xyz = (cur_xyz - cur_xyz.max(dim=0)[0]) / (cur_xyz.max(dim=0)[0] - cur_xyz.min(dim=0)[0] + 1e-6)
        motion_weight = self.MotionWeighting(norm_xyz)
        motion_weight = F.softmax(motion_weight, dim=1)

        extracted_features = self.GaussianFeatureExtractor(cur_fea) # N, latent_dim
        extracted_features = position_features + extracted_features
        ctx_params_motion = extracted_features
        # ctx_params_motion = extracted_features.view(1, self.lat_dim, -1) # [1, latent_dim, N]


        # Q_z = self.AdaptiveQuantizationZ(ctx_params_motion) * 2 * self.Q_z # [N_m, latent_dim]
        # Q_z = Q_z.squeeze(0).permute(1, 0).contiguous()
        Q_z = self.Q_z
        z_hat_motion = decoder_factorized_chunk(self.EntropyFactorizedMotion._logits_cumulative, Q_z, N_gaussian, self.lat_dim, z_hat_bit_path, chunk_size=10000)
        # z_hat_motion = z_hat_motion.unsqueeze(0).permute(0, 2, 1).contiguous() # [1, latent_dim, N_m]
        params_motion = self.MotionPriorDecoder(z_hat_motion) # [1, latent_dim, N_m]

        # distribution_motion = self.EntropyParametersMotion(params_motion)
        distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1))
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6

        Q_y = self.Q_y
        # Q_y = (self.AdaptiveQuantizationY(ctx_params_motion.detach()) + 1) * self.Q_y 
        # Q_y = Q_y.squeeze(0).permute(1, 0).contiguous().repeat(1, self.lat_dim)
        # Q_y = Q_y.mean()
        y_hat_motion = decoder_gaussian_chunk(mean_motion.contiguous(), std_motion, Q_y, y_hat_bit_path, chunk_size=100000) # [N_m, latent_dim]
        # y_hat_motion = y_hat_motion.view(N_gaussian, -1).unsqueeze(0).permute(0, 2, 1).contiguous() # [1, latent_dim, N_m]
        y_hat_motion = y_hat_motion.view(N_gaussian, -1)

        # mask_motion = decoder(y_hat_motion.shape[0], mask_bit_path)
        # mask_motion = mask_motion.to(torch.bool).squeeze()
        # mask_motion = torch.ones_like(mask_motion, dtype=torch.bool).detach()

        # dec_motion = torch.zeros(N_gaussian, self.motion_dim, device=cur_xyz.device, dtype=cur_xyz.dtype)
        # dec_motion[mask_motion] = self.MotionDecoder(torch.cat([y_hat_motion, cur_xyz], dim=1))
        dec_motion = self.MotionDecoder(y_hat_motion)

        # 绘制到同一张直方图并保存
        import matplotlib.pyplot as plt
        import pandas as pd

        data_y_hat_motion = y_hat_motion.view(-1).detach().cpu().numpy()
        data_dec_motion = dec_motion.view(-1).detach().cpu().numpy()
        # plt.hist(data_y_hat_motion, bins=2000, alpha=0.5, label='y_hat_motion')  
        
        plt.hist(data_dec_motion, bins=2000, alpha=0.5, label='dec_motion')
        
        # 打印data_dec_motion中不同的值
        print(pd.Series(data_dec_motion.flatten()).value_counts())
        print(pd.Series(data_y_hat_motion.flatten()).value_counts())


        plt.xlim(-0.5, 0.5)
        
        # print(data_y_hat_motion.max(), data_y_hat_motion.min())
        plt.legend()
        plt.title('Histogram of y_hat_motion and dec_motion')
        
        save_path = f'{y_hat_bit_path.split(".b")[0]}.png'
        plt.savefig(save_path)
        plt.clf()

        
        # dec_motion = dec_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, motion_dim]
        # dec_motion = dec_motion * motion_weight
        
        nxt_gaussians = self.MotionCompensation(dynamicGS_type, dec_motion, cur_gaussians)

        # Residual Compensation
        residual_ctx_params = extracted_features # [N, latent_dim]
        residual_feature = self.ResidualGenerator(residual_ctx_params)
        nxt_gaussians = self.ResidualCompensation(nxt_gaussians, residual_feature=residual_feature)

        print('y_hat_motion: ', y_hat_motion)
        print('dec_motion: ', dec_motion)
        print(y_hat_motion.sum())

        return nxt_gaussians

    def compress_I(self):
        pass

    def decompress_I(self):
        pass

    def forward_I(self):
        pass

    def test(self):
        for i in range(len(self.scene.getTestCameras().copy())):
            loss = self.ComputeRenderLoss(self.nxt_gaussians, False)
            print(f'loss_{i}: ', loss)

    def forward(self):
        
        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(self.cur_gaussians)
        # print(N_gaussian)
        cur_xyz, cur_fea = sorted_orig_voxels(cur_xyz, cur_fea)
        
        # norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(cur_xyz, K=self.norm_radius)
        # shuffled_indices = torch.randperm(cur_xyz.shape[0], device=cur_xyz.device)
        # cur_xyz = cur_xyz[shuffled_indices]
        # cur_fea = cur_fea[shuffled_indices]

        # original_indices = torch.zeros_like(shuffled_indices)
        # original_indices[shuffled_indices] = torch.arange(shuffled_indices.size(0), device=shuffled_indices.device)

        # sampled_xyz, idx = sample_farthest_points(cur_xyz.unsqueeze(0), K=N_gaussian//100)
        idx = torch.randint(0, N_gaussian, (N_gaussian//2,), device=cur_xyz.device)

        position_features = self.GridEncoder(cur_xyz)

        # normalize the position_features
        position_features = F.normalize(position_features, p=2, dim=1)

        norm_xyz = (cur_xyz - cur_xyz.max(dim=0)[0]) / (cur_xyz.max(dim=0)[0] - cur_xyz.min(dim=0)[0] + 1e-6)
        motion_weight = self.MotionWeighting(norm_xyz)
        motion_weight = F.softmax(motion_weight, dim=1)
        # print('motion_weighting: ', motion_weight)
        # print('motion_weighting: ', motion_weight.sum(dim=1, keepdim=True))
        # print('motion_weighting: ', motion_weight.shape)

        # print('motion_weight: ', motion_weight.sum(dim=1, keepdim=True))

        extracted_features = self.GaussianFeatureExtractor(cur_fea) # N, latent_dim
        extracted_features = position_features + extracted_features

        # print(position_features)
        # print(extracted_features)

        # mask_motion_tmp = self.MotionMaskGenerator(extracted_features.detach().to(torch.float32)) # [N, 1]
        # mask_motion = ((mask_motion_tmp > 0.001).float() - mask_motion_tmp).detach() + mask_motion_tmp  # [N, 1]
        # mask_motion = mask_motion.to(torch.bool).squeeze()

        ctx_params_motion = extracted_features
        # ctx_params_motion = extracted_features.view(1, self.lat_dim, -1) # [1, latent_dim, N]
        
        # TODO context in 3D Space
        # ctx_params_motion = self.AutoRegressiveMotion(extracted_features) # [1, latent_dim, N]

        est_motion = self.MotionEstimation(self.args.dynamicGS_type, self.cur_gaussians, self.nxt_gaussians) # I-NGP in 3DGStream / MLP in Deformable-GS / Hexplane in 4D-GS
        # est_motion = est_motion[shuffled_indices]
        print('est_motion: ', est_motion)
        key_est_motion = est_motion[idx, :]
        # print('key_est_motion: ', key_est_motion)

        # mask_motion = torch.ones(mask_motion, dtype=torch.bool).detach()

        # est_motion = est_motion[mask_motion] # [N_m, motion_dim]
        # motion_weight = motion_weight[mask_motion]

        # NOTE: we use y denotes the latent representation, z denotes the hyperprior, and hat denotes the quantized version
        # est_motion = est_motion.unsqueeze(0).permute(0, 2, 1).contiguous() # [1, motion_dim, N_m]
        y_motion = self.MotionEncoder(est_motion) # [1, latent_dim, N_m]
        

        Q_y = self.Q_y
        # Q_y = (self.AdaptiveQuantizationY(ctx_params_motion.detach()) + 1) * self.Q_y 
        
        y_hat_motion = self.quantize(y_motion, Q=Q_y, train_flag=False) # [1, latent_dim, N_m]

        z_motion = self.MotionPriorEncoder(y_motion) # [1, latent_dim, N_m]
        Q_z = self.Q_z
        # Q_z = self.AdaptiveQuantizationZ(ctx_params_motion) * 2 * self.Q_z

        z_hat_motion = self.quantize(z_motion, Q=Q_z, train_flag=False) # [1, latent_dim, N_m]
        params_motion = self.MotionPriorDecoder(z_hat_motion) # [1, latent_dim, N_m]

        # Motion Decoding
        dec_motion = self.MotionDecoder(y_hat_motion) # [1, motion_dim, N_m]
        # dec_motion = dec_motion[original_indices]
        # dec_motion = dec_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, motion_dim]

        # print(motion_weight[1:3, :])
        # dec_motion = dec_motion * motion_weight

        print('y_hat_motion: ', y_hat_motion)
        print('y_hat_motion: ', y_hat_motion.sum())
        print('dec_motion: ', dec_motion)

        key_dec_motion = dec_motion[idx, :]
        # print('key_dec_motion: ', key_dec_motion)

        # Motion Compensation
        nxt_gaussians = self.MotionCompensation(self.args.dynamicGS_type, dec_motion, self.cur_gaussians)
        with torch.no_grad():
            self.buffer = nxt_gaussians

        # Residual Compensation
        residual_ctx_params = extracted_features # [N, latent_dim]
        residual_feature = self.ResidualGenerator(residual_ctx_params)
        # residual_feature = residual_feature[original_indices]
        nxt_gaussians = self.ResidualCompensation(nxt_gaussians, residual_feature=residual_feature)

        # Entropy Calculation
        distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1)) 

        # distribution_motion = distribution_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, 2*latent_dim]
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1) # [N_m, latent_dim]
        std_motion = F.softplus(std_motion) + 1e-6 # [N_m, latent_dim]
        mean_motion = mean_motion.contiguous()

        # y_hat_motion = y_hat_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, latent_dim]
        # z_hat_motion = z_hat_motion.squeeze(0).permute(1, 0).contiguous() # [N_m, latent_dim]
        # Q_y = Q_y.squeeze(0).permute(1, 0).contiguous().repeat(1, self.lat_dim) # [N_m, latent_dim]

        bits_motion = self.EntropyGaussianMotion(y_hat_motion, mean_motion, std_motion, self.Q_y)
        bits_prior_motion = self.EntropyFactorizedMotion(z_hat_motion)

        # loss_mask = torch.mean(mask_motion_tmp)
        loss_mask = torch.tensor(0.0, device=dec_motion.device)
        total_size = bits_motion.mean() + bits_prior_motion.mean()  
        loss_render = self.ComputeRenderLoss(nxt_gaussians)
        loss_motion_mean = F.mse_loss(key_est_motion.detach(), key_dec_motion)
        # loss_motion_mean = torch.tensor(0.0, device=dec_motion.device)
        return loss_render, total_size, loss_mask, loss_motion_mean
        # return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)