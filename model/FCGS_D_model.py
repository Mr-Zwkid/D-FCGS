import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import copy
from scene import GaussianModel, Scene
from model.encodings_cuda import STE_multistep, encoder_gaussian, decoder_gaussian, encoder_gaussian_chunk, decoder_gaussian_chunk, \
    encoder_gaussian_mixed, decoder_gaussian_mixed, encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk,\
    encoder_factorized, decoder_factorized, encoder_factorized_chunk, decoder_factorized_chunk
from model.entropy_models import Entropy_factorized, Entropy_gaussian
from model.subnet import GDN1D, MaskedConv1d
from random import randint
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from model.motion_estimators import NeuralTransformationCache

import pickle

bit_to_MB = 8 * 1024 * 1024

def GaussianParameterPack(gaussians):
    g_xyz = gaussians._xyz.detach() # [N, 3]
    N_gaussian = g_xyz.shape[0] # N
    _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
    _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
    _opacity = gaussians._opacity.detach()  # [N, 1]
    _scaling = gaussians._scaling.detach()  # [N, 3]
    _rotation = gaussians._rotation.detach()  # [N, 4]
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
        self.gof_size = args.gof_size
        self.gaussian_position_dim = 3
        self.gaussian_feature_dim = args.gaussian_feature_dim # 56
        self.motion_dim = args.motion_dim # 7
        self.hidden_dim = args.hidden_dim # 256
        self.lat_dim = args.lat_dim # 256
        self.GDN = GDN1D

        self.cur_gaussians = self.read_gaussian_file(args.init_3dgs, sh_degree=3)
        self.nxt_gaussians = None
        self.scene = Scene(args)
        self.viewpoint_stack = None
        # self.init_test_gaussians()

        self.MotionEstimator = None
        self.MotionEstimatorSetup(self.args.dynamicGS_type, self.args.motion_estimator_path)

        self.GaussianFeatureExtractor = nn.Sequential(
            nn.Linear(self.gaussian_feature_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim)
        )

        self.MotionMaskGenerator = nn.Sequential(
            # nn.Linear(self.gaussian_feature_dim, 1),
            nn.Linear(self.gaussian_feature_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

        self.FeatureExtractor = nn.Sequential(
            nn.Linear(self.gaussian_feature_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim)
        )

        self.MotionEncoder = nn.Sequential(
            nn.Linear(self.motion_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim),
        )

        self.MotionDecoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.motion_dim),
        )

        self.MotionPriorEncoder = nn.Sequential(
            nn.Linear(self.motion_dim, self.hidden_dim),
            # nn.Linear(self.lat_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim),
        )

        self.MotionPriorDecoder = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.GDN(self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.lat_dim),
        )

        self.AutoRegressiveMotion = nn.Sequential(
            MaskedConv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, mask_type="A"),
            nn.LeakyReLU(inplace=True),
            MaskedConv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, mask_type="A"),
            nn.LeakyReLU(inplace=True),
            MaskedConv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, mask_type="A"),
            nn.LeakyReLU(inplace=True)
        )

        self.EntropyParametersMotion = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            # nn.Linear(self.hidden_dim, 2 * self.lat_dim)
            nn.Linear(self.hidden_dim, 2 * self.motion_dim)
        )

        self.EntropyFactorizedMotion = Entropy_factorized(self.lat_dim, Q=self.Q_z)

        self.EntropyGaussianMotion = Entropy_gaussian(Q=self.Q_y)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # 获取完整 state_dict
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # 删除不需要保存的项，这里假设所有以 "MotionEstimator"、“cur_gaussians” 或 “nxt_gaussians” 开头的键删除
        keys_to_remove = [key for key in state.keys() if key.startswith('MotionEstimator') 
                                              or key.startswith('cur_gaussians') 
                                              or key.startswith('nxt_gaussians')]
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

    def ComputeRenderLoss(self, gaussians):

        bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_camera = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_camera, gaussians, self.args, background)
        rendered_img = render_pkg["render"]
        # TODO use depth or optical flow as prior

        gt_img = viewpoint_camera.original_image.cuda()
        L1_loss = l1_loss(rendered_img, gt_img)
        ssim_loss = ssim(rendered_img, gt_img)
        render_loss = (1.0 - self.args.lambda_dssim) * L1_loss + self.args.lambda_dssim * (1.0 - ssim_loss)
        return render_loss

    def MotionEstimatorSetup(self, dynamicGS_type, motion_estimator_path=None):
        if dynamicGS_type == '3dgstream':
            self.MotionEstimator = NeuralTransformationCache(*self.get_xyz_bound(self.cur_gaussians))
        # elif dynamicGS_type == 'deformable-gs':
        #     self.MotionEstimator = NeuralTransformationCache()
        # elif dynamicGS_type == '4d-gs':
        #     self.MotionEstimator = NeuralTransformationCache()
        else:
            raise ValueError('Invalid dynamicGS_type')

        self.MotionEstimator.load_state_dict(torch.load(motion_estimator_path), strict=True)
        self.MotionEstimator.requires_grad_(False)
        
    def MotionEstimation(self, dynamicGS_type, xyz):
        if dynamicGS_type == '3dgstream':
            mask, d_xyz, d_rot = self.MotionEstimator(xyz)
            return torch.cat((d_xyz, d_rot), dim=1)
        
    def MotionCompensation(self, dynamicGS_type, dec_motion, cur_gaussians, mask_motion=None):
        # nxt_gaussians = copy.deepcopy(cur_gaussians)
        nxt_gaussians = copy.deepcopy(cur_gaussians)
        if dynamicGS_type == '3dgstream':
            # dec_motion[~mask_motion] = 0
            nxt_gaussians._xyz = nn.Parameter(cur_gaussians._xyz.detach() + dec_motion[:, :3])
            nxt_gaussians._rotation = nn.Parameter(quaternion_multiply(cur_gaussians._rotation.detach(),  dec_motion[:, 3:]))
            
        return nxt_gaussians
            
    def refresh_gaussians(self, cur_gaussian_path, nxt_gaussian_path, sh_degree = 3):
        self.cur_gaussians = self.nxt_gaussians

    def read_gaussian_file(self, file_path, sh_degree = 3):
        with torch.no_grad():
            gaussians = GaussianModel(sh_degree)
            gaussians.load_ply(file_path)
        return gaussians
    
    def init_test_gaussians(self, sh_degree = 3):
        self.cur_gaussians = self.read_gaussian_file('/SDD_D/zwk/output/cook_spinach-3-ori/init_3dgs.ply')
        self.nxt_gaussians = self.read_gaussian_file('/SDD_D/zwk/output/cook_spinach-3-ori/init_3dgs.ply')

    def compress(self, cur_gaussians_path, motion_estimator_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', dynamicGS_type='3dgstream'):
        
        cur_gaussians = self.read_gaussian_file(cur_gaussians_path)
        self.MotionEstimatorSetup(dynamicGS_type, motion_estimator_path)

        # mask for motion
        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(cur_gaussians)
        mask_motion_tmp = self.MotionMaskGenerator(cur_fea) # N, 1
        mask_motion = ((mask_motion_tmp > 0.01).float() - mask_motion_tmp).detach() + mask_motion_tmp  # [N, 1]
        mask_motion = mask_motion.to(torch.bool).squeeze()

        mask_motion = torch.ones_like(mask_motion)

        extracted_features = self.GaussianFeatureExtractor(cur_fea[mask_motion]) # N, latent_dim
        est_motion = self.MotionEstimation(dynamicGS_type, cur_gaussians._xyz[mask_motion]) # I-NGP in 3DGStream / MLP in Deformable-GS / Hexplane in 4D-GS

        # y_motion = self.MotionEncoder(est_motion.to(torch.float32)) # N, latent_dim
        y_motion = est_motion.to(torch.float32)
        y_hat_motion = self.quantize(y_motion, Q=self.Q_y, train_flag=False)

        # ctx_params_motion = self.AutoRegressiveMotion(extracted_features.reshape(-1, 1, self.lat_dim)).view(-1, self.lat_dim) # N, latent_dim
        z_motion = self.MotionPriorEncoder(y_motion) # N, latent_dim
        z_hat_motion = self.quantize(z_motion, Q=self.Q_z, train_flag=False)
        params_motion = self.MotionPriorDecoder(z_hat_motion) # N, latent_dim

        # distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1)) # N, 2*latent_dim
        distribution_motion = self.EntropyParametersMotion(params_motion) # N, 2*latent_dim
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6

        bits_motion = encoder_gaussian_chunk(y_hat_motion, mean_motion.contiguous(), std_motion, 1e-1, y_hat_bit_path)
        bits_prior_motion = encoder_factorized_chunk(z_hat_motion, self.EntropyFactorizedMotion._logits_cumulative, self.Q_z, z_hat_bit_path)

        return {
            'bits_motion': bits_motion / bit_to_MB,
            'bits_prior_motion': bits_prior_motion / bit_to_MB,
        }

    def decompress(self, cur_gaussians_path, y_hat_bit_path, z_hat_bit_path, dynamicGS_type='3dgstream'):
        cur_gaussians = self.read_gaussian_file(cur_gaussians_path)

        # mask for motion
        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(cur_gaussians)
        mask_motion_tmp = self.MotionMaskGenerator(cur_fea)
        mask_motion = ((mask_motion_tmp > 0.01).float() - mask_motion_tmp).detach() + mask_motion_tmp
        mask_motion = mask_motion.to(torch.bool).squeeze()

        mask_motion = torch.ones_like(mask_motion)

        extracted_features = self.GaussianFeatureExtractor(cur_fea[mask_motion]) # N, latent_dim

        # ctx_params_motion = self.AutoRegressiveMotion(extracted_features.reshape(-1, 1, self.lat_dim)).view(-1, self.lat_dim)

        z_hat_motion = decoder_factorized_chunk(self.EntropyFactorizedMotion._logits_cumulative, self.Q_z, cur_fea[mask_motion].shape[0], self.lat_dim, z_hat_bit_path)
        params_motion = self.MotionPriorDecoder(z_hat_motion)

        distribution_motion = self.EntropyParametersMotion(params_motion)
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6

        y_hat_motion = decoder_gaussian_chunk(mean_motion.contiguous(), std_motion, self.Q_y, y_hat_bit_path)

        y_hat_motion = y_hat_motion.reshape(-1, self.motion_dim)
        dec_motion = y_hat_motion
        # y_hat_motion = y_hat_motion.reshape(-1, self.lat_dim)
        # dec_motion = torch.zeros(N_gaussian, self.motion_dim, device=cur_xyz.device, dtype=cur_xyz.dtype)
        # dec_motion[mask_motion] = self.MotionDecoder(y_hat_motion)
        nxt_gaussians = self.MotionCompensation(dynamicGS_type, dec_motion, cur_gaussians, mask_motion)
        return nxt_gaussians
    
    def forward(self):
        # mask for motion
        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(self.cur_gaussians)

        mask_motion_tmp = self.MotionMaskGenerator(cur_fea) # N, 1
        mask_motion = ((mask_motion_tmp > 0.01).float() - mask_motion_tmp).detach() + mask_motion_tmp  # [N, 1]
        mask_motion = mask_motion.to(torch.bool).squeeze()

        mask_motion = torch.ones_like(mask_motion)

        # with torch.amp.autocast('cuda'):

        # extracted_features = self.GaussianFeatureExtractor(cur_fea[mask_motion]) # N, latent_dim

        est_motion = self.MotionEstimation(self.args.dynamicGS_type, self.cur_gaussians._xyz[mask_motion]) # I-NGP in 3DGStream / MLP in Deformable-GS / Hexplane in 4D-GS

        # NOTE: we use y denotes the latent representation, z denotes the hyperprior, and hat denotes the quantized version
        # y_motion = self.MotionEncoder(est_motion.to(torch.float32)) # N, latent_dim
        y_motion = est_motion.to(torch.float32)

        y_hat_motion = self.quantize(y_motion, Q=self.Q_y, train_flag=True)

        # TODO 在空间上进行context的计算
        # ctx_params_motion = self.AutoRegressiveMotion(extracted_features.reshape(-1, 1, self.lat_dim)).view(-1, self.lat_dim) # N, latent_dim

        z_motion = self.MotionPriorEncoder(y_motion) # N, latent_dim
        z_hat_motion = self.quantize(z_motion, Q=self.Q_z, train_flag=True)
        params_motion = self.MotionPriorDecoder(z_hat_motion) # N, latent_dim

        # distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1)) # N, 2*latent_dim
        distribution_motion = self.EntropyParametersMotion(params_motion) # N, 2*latent_dim
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6

        bits_motion = self.EntropyGaussianMotion(y_hat_motion, mean_motion, std_motion, self.Q_y)
        bits_prior_motion = self.EntropyFactorizedMotion(z_hat_motion)

        # Motion Decoding
        # dec_motion = self.MotionDecoder(y_hat_motion)
        dec_motion = y_hat_motion

        if dec_motion.shape[0] != mask_motion.shape[0]:
            diff = mask_motion.shape[0] - dec_motion.shape[0]
            if diff > 0:
                pad = torch.zeros(diff, dec_motion.shape[1], device=dec_motion.device, dtype=dec_motion.dtype)
                dec_motion = torch.cat([dec_motion, pad], dim=0)
            else:
                dec_motion = dec_motion[:mask_motion.shape[0], :]

        # Motion Compensation
        self.nxt_gaussians = self.MotionCompensation(self.args.dynamicGS_type, dec_motion, self.cur_gaussians, mask_motion)
        # print(f'cur_gaussians: {self.cur_gaussians._xyz}, nxt_gaussians: {self.nxt_gaussians._xyz}')

        loss_render = self.ComputeRenderLoss(self.nxt_gaussians)
        # loss_mask = torch.mean(mask_motion_tmp)
        loss_mask = torch.tensor(0.0, device=mask_motion.device)
        total_size = bits_motion.mean() + bits_prior_motion.mean()
        return loss_render, total_size, loss_mask
        