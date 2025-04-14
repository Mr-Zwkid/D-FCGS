import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import copy
from scene import GaussianModel, Scene, SimpleGaussianModel
from model.gpcc_utils import sorted_voxels, sorted_orig_voxels, compress_gaussian_params, decompress_gaussian_params
from model.encodings_cuda import STE_multistep, encoder_gaussian_chunk, decoder_gaussian_chunk, encoder_factorized_chunk, decoder_factorized_chunk, encoder, decoder
from model.entropy_models import Entropy_factorized, Entropy_gaussian
from model.subnet import GDN1D, MaskedConv1d, DownsampleLayer, ClipLayer
from random import randint
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from model.motion_estimators import NeuralTransformationCache
from model.grid_utils import normalize_xyz, _grid_creater, _grid_encoder, FreqEncoder
from pytorch3d.ops import sample_farthest_points, knn_points

bit_to_MB = 8 * 1024 * 1024

def plot_hist(x, title='histogram', bins=100, range=None, save_path=None):
    import matplotlib.pyplot as plt
    plt.clf()
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
        self.gof_size = args.gof_size
        self.gaussian_position_dim = 3
        self.gaussian_feature_dim = args.gaussian_feature_dim
        self.motion_dim = args.motion_dim
        self.hidden_dim = args.hidden_dim
        self.lat_dim = args.lat_dim 
        self.motion_limit = args.motion_limit
        self.downsample_rate = args.downsample_rate
        self.knn_num = args.knn_num

        self.without_refinement = args.without_refinement
        self.without_context = args.without_context
        self.without_hyper = args.without_hyper

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

        self.FramePositionExtractor = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_freq'], network_config=self.GridEncoder_config['network'])
        self.FrameRotationExtractor = tcnn.NetworkWithInputEncoding(n_input_dims=4, n_output_dims=self.lat_dim, encoding_config=self.GridEncoder_config['encoding_freq'], network_config=self.GridEncoder_config['network'])
      
        self.Combiner = nn.Sequential(
            nn.Linear(2 * self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.motion_dim)
        )

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
            ClipLayer(-self.motion_limit, self.motion_limit)
        )

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
        
        self.EntropyParametersMotion = nn.Sequential(
            nn.Linear(2 * self.lat_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2 * self.lat_dim)
        )

        self.EntropyParametersMotion_NO_CONTEXT = nn.Sequential(
            nn.Linear(self.lat_dim, self.hidden_dim),
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
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
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
            def get_xyz_bound(self, gaussians, percentile=86.6):
                half_percentile = (100 - percentile) / 200
                xyz_bound_min = torch.quantile(gaussians._xyz,half_percentile,dim=0)
                xyz_bound_max = torch.quantile(gaussians._xyz,1 - half_percentile,dim=0)
                return xyz_bound_min, xyz_bound_max
            self.MotionEstimator = NeuralTransformationCache(*self.get_xyz_bound(self.cur_gaussians))
            self.MotionEstimator.load_state_dict(torch.load(motion_estimator_path), strict=True)
            self.MotionEstimator.requires_grad_(False)
        elif dynamicGS_type == '3dgstream_explicit':
            pass
        elif dynamicGS_type == '3dgstream_implicit':
            pass
        elif dynamicGS_type == 'control_point':
            pass
        else:
            raise ValueError('Invalid dynamicGS_type')
        
    def MotionEstimation(self, dynamicGS_type, cur_gaussians, nxt_gaussians=None):
        if dynamicGS_type == '3dgstream':
            xyz = cur_gaussians._xyz
            mask, d_xyz, d_rot = self.MotionEstimator(xyz)
            return torch.cat((d_xyz, d_rot), dim=1), None
        
        elif dynamicGS_type == '3dgstream_explicit':
            cur_xyz = cur_gaussians._xyz
            nxt_xyz = nxt_gaussians._xyz

            cur_rot = cur_gaussians._rotation
            nxt_rot = nxt_gaussians._rotation

            assert cur_xyz.shape[0] == nxt_xyz.shape[0], 'The number of points in the current and next frame should be the same'
            
            # NOTE avoid the rotation flip
            if abs(cur_rot.mean() + nxt_rot.mean()) < abs(cur_rot.mean()):
                nxt_rot = -nxt_rot
            
            d_xyz = nxt_xyz - cur_xyz
            d_rot = nxt_rot - cur_rot

            return torch.cat((d_xyz, d_rot), dim=1), None
        
        elif dynamicGS_type == '3dgstream_implicit':
            cur_xyz = cur_gaussians._xyz.detach()
            nxt_xyz = nxt_gaussians._xyz.detach()

            # Min-max normalization for position
            cur_xyz = (cur_xyz - cur_xyz.min(dim=0)[0]) / (cur_xyz.max(dim=0)[0] - cur_xyz.min(dim=0)[0] + 1e-6)
            nxt_xyz = (nxt_xyz - nxt_xyz.min(dim=0)[0]) / (nxt_xyz.max(dim=0)[0] - nxt_xyz.min(dim=0)[0] + 1e-6)
            cur_xyz_trans = self.FramePositionExtractor(cur_xyz)
            nxt_xyz_trans = self.FramePositionExtractor(nxt_xyz)

            cur_rot = cur_gaussians._rotation.detach()
            nxt_rot = nxt_gaussians._rotation.detach()

            # NOTE avoid the rotation flip
            if abs(cur_rot.mean() + nxt_rot.mean()) < abs(cur_rot.mean()):
                nxt_rot = -nxt_rot

            cur_rot = (cur_rot - cur_rot.mean(dim=0)[0]) / (cur_rot.max(dim=0)[0] - cur_rot.min(dim=0)[0] + 1e-6)
            nxt_rot = (nxt_rot - nxt_rot.mean(dim=0)[0]) / (nxt_rot.max(dim=0)[0] - nxt_rot.min(dim=0)[0] + 1e-6)
            cur_rot_trans = self.FrameRotationExtractor(cur_rot)
            nxt_rot_trans = self.FrameRotationExtractor(nxt_rot)

            motion_xyz = (nxt_xyz_trans - cur_xyz_trans)
            motion_rot = (nxt_rot_trans - cur_rot_trans)
            motion = self.Combiner(torch.cat([motion_xyz, motion_rot], dim=1).to(torch.float32))

            return motion, None
        
        elif dynamicGS_type == 'control_point':
            
            cur_xyz = cur_gaussians._xyz.detach()
            nxt_xyz = nxt_gaussians._xyz.detach()
            sample_num = min(cur_xyz.shape[0] // self.downsample_rate, nxt_xyz.shape[0] // self.downsample_rate) 

            sampled_cur_xyz, idx_cur = sample_farthest_points(cur_xyz.unsqueeze(0), K=sample_num)
            sampled_nxt_xyz, idx_nxt = sample_farthest_points(nxt_xyz.unsqueeze(0), K=sample_num)

            sampled_cur_xyz = sampled_cur_xyz.squeeze(0)
            sampled_nxt_xyz = sampled_nxt_xyz.squeeze(0)

            cur_rot = cur_gaussians._rotation.detach()
            nxt_rot = nxt_gaussians._rotation.detach()
            # NOTE avoid the rotation flip
            if abs(cur_rot.mean() + nxt_rot.mean()) < abs(cur_rot.mean()):
                nxt_rot = -nxt_rot

            cur_rot = (cur_rot - cur_rot.mean(dim=0)[0]) / (cur_rot.max(dim=0)[0] - cur_rot.min(dim=0)[0] + 1e-6)
            nxt_rot = (nxt_rot - nxt_rot.mean(dim=0)[0]) / (nxt_rot.max(dim=0)[0] - nxt_rot.min(dim=0)[0] + 1e-6)

            sampled_cur_rot = cur_rot[idx_cur].squeeze(0)
            sampled_nxt_rot = nxt_rot[idx_nxt].squeeze(0)
            # print('sampled_cur_xyz: ', sampled_cur_xyz)
            # print('sampled_nxt_xyz: ', sampled_nxt_xyz)
            # print('sampled_cur_rot: ', sampled_cur_rot)
            # print('sampled_nxt_rot: ', sampled_nxt_rot)

            cur_xyz_trans = self.FramePositionExtractor(sampled_cur_xyz)
            nxt_xyz_trans = self.FramePositionExtractor(sampled_nxt_xyz)

            cur_rot_trans = self.FrameRotationExtractor(sampled_cur_rot)
            nxt_rot_trans = self.FrameRotationExtractor(sampled_nxt_rot)

            motion_xyz = (nxt_xyz_trans - cur_xyz_trans)
            motion_rot = (nxt_rot_trans - cur_rot_trans)
            # print('motion_xyz: ', motion_xyz)
            # print('motion_rot: ', motion_rot)

            motion = self.Combiner(torch.cat([motion_xyz, motion_rot], dim=1).to(torch.float32))
            # print('motion: ', motion)
            
            return motion, idx_cur
    
    def MotionCompensation(self, dynamicGS_type, dec_motion, cur_gaussians, nxt_gaussians_=None, idx_cur = None):
        nxt_gaussians = SimpleGaussianModel(cur_gaussians)
        if dynamicGS_type == '3dgstream':
            nxt_gaussians._xyz = cur_gaussians._xyz.detach() + dec_motion[:, :3]
            nxt_gaussians._rotation = quaternion_multiply(cur_gaussians._rotation.detach(), dec_motion[:, 3:])

        elif dynamicGS_type == '3dgstream_explicit':
            nxt_gaussians._xyz = cur_gaussians._xyz.detach() + dec_motion[:, :3]
            nxt_gaussians._rotation = cur_gaussians._rotation.detach() + dec_motion[:, 3:]

        elif dynamicGS_type == '3dgstream_implicit':
            nxt_gaussians._xyz = cur_gaussians._xyz.detach() + dec_motion[:, :3]
            nxt_gaussians._rotation = cur_gaussians._rotation.detach() + dec_motion[:, 3:]
            
        elif dynamicGS_type == 'control_point':
            cur_xyz = cur_gaussians._xyz.detach()
            cur_rot = cur_gaussians._rotation.detach()

            query_xyz = cur_xyz[idx_cur]
            knn_xyz = knn_points(query_xyz, cur_xyz.unsqueeze(0), K=self.knn_num)
            dists = knn_xyz.dists.squeeze(0)
            knn_idx = knn_xyz.idx.squeeze(0)
            softmax_dists = F.softmax(-dists, dim=1).unsqueeze(-1)

            dec_motion = dec_motion.unsqueeze(1)

            dec_motion_expanded = dec_motion * softmax_dists
            # plot_hist(dec_motion_expanded, title='dec_motion_expanded', bins=100, range=(-0.5, 0.5), save_path='./dec_motion_expanded.png')

            # for every line of knn_idx, add the corresponding dec_motion
            nxt_gaussians._xyz = cur_xyz
            nxt_gaussians._rotation = cur_rot
            for i in range(knn_idx.shape[0]):
                nxt_gaussians._xyz[knn_idx[i]] = nxt_gaussians._xyz[knn_idx[i]] + dec_motion_expanded[i][:, :3]
                nxt_gaussians._rotation[knn_idx[i]] = nxt_gaussians._rotation[knn_idx[i]] + dec_motion_expanded[i][:, 3:]
                # print('i: ', i)
                # print('knn_idx[i]: ', knn_idx[i])
                # print('dec_motion_expanded[i]: ', dec_motion_expanded[i].shape)
                # print('nxt_gaussians._xyz[knn_idx[i]]: ', nxt_gaussians._xyz[knn_idx[i]].shape)

        return nxt_gaussians

    def Refinement(self, cur_gaussians, residual_position=None, residual_feature=None): 
        if self.without_refinement:
            return cur_gaussians
        
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
        if args.next_3dgs is not None:
            self.nxt_gaussians = self.read_gaussian_file(args.next_3dgs, sh_degree=3)
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

    def compress(self, cur_gaussians_path, motion_estimator_path, y_hat_bit_path = 'motion.b', z_hat_bit_path = 'motion_prior.b', mask_bit_path = 'mask.b', dynamicGS_type='3dgstream', nxt_gaussians_path=None, use_buffer = False, buffer_gaussian = None):
        
        cur_gaussians = self.read_gaussian_file(cur_gaussians_path) if not use_buffer else buffer_gaussian
        nxt_gaussians = self.read_gaussian_file(nxt_gaussians_path) if nxt_gaussians_path is not None else None
        self.MotionEstimatorSetup(dynamicGS_type, motion_estimator_path) # useless

        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(cur_gaussians)

        position_features = self.GridEncoder(cur_xyz)
        position_features = F.normalize(position_features, p=2, dim=1)
        extracted_features = self.GaussianFeatureExtractor(cur_fea)
        ctx_params_motion = position_features + extracted_features
        
        est_motion, idx_cur = self.MotionEstimation(dynamicGS_type, cur_gaussians, nxt_gaussians)
        
        y_motion = self.MotionEncoder(est_motion)
        y_hat_motion = self.quantize(y_motion, Q=self.Q_y, train_flag=False)

        z_motion = self.MotionPriorEncoder(y_motion) 
        z_hat_motion = self.quantize(z_motion, Q=self.Q_z, train_flag=False)
        params_motion = self.MotionPriorDecoder(z_hat_motion)

        ctx_params_motion = ctx_params_motion[idx_cur].squeeze(0) if idx_cur is not None else ctx_params_motion
        distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1)) if not self.without_context else self.EntropyParametersMotion_NO_CONTEXT(params_motion)
 
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6
        mean_motion = mean_motion.contiguous()

        plot_hist(est_motion, save_path=y_hat_bit_path.replace('.b', '_est_motion.png'), title='est_motion', range=(-0.3, 0.3), bins=100)
        # print('est_motion: ', est_motion, est_motion.max(), est_motion.min())
        # print('y_motion: ', y_motion)  
        # print('y_hat_motion: ', y_hat_motion, y_hat_motion.sum())
        # print('z_hat_motion: ', z_hat_motion)

        # dec_motion = self.MotionDecoder(y_hat_motion) # [1, motion_dim, N_m]
        # print('dec_motion: ', dec_motion, dec_motion.max(), dec_motion.min(), dec_motion.sum(dim=1))
        # print('mse: ', F.mse_loss(dec_motion, est_motion))

        bits_motion = encoder_gaussian_chunk(y_hat_motion, mean_motion.contiguous(), std_motion, Q_y, y_hat_bit_path, chunk_size=100000)
        bits_prior_motion = encoder_factorized_chunk(z_hat_motion, self.EntropyFactorizedMotion._logits_cumulative, Q_z, z_hat_bit_path, chunk_size=10000)

        return {
            'bits_motion': bits_motion / bit_to_MB,
            'bits_prior_motion': bits_prior_motion / bit_to_MB,
        }

    def decompress(self, cur_gaussians_path, y_hat_bit_path, z_hat_bit_path, mask_bit_path, dynamicGS_type='3dgstream'):
        
        cur_gaussians = self.read_gaussian_file(cur_gaussians_path)

        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(self.cur_gaussians)

        if dynamicGS_type == 'control_point':
            sampled_num = N_gaussian // self.downsample_rate
            sampled_cur_xyz, idx_cur = sample_farthest_points(cur_xyz.unsqueeze(0), K=sampled_num)
        else :
            sampled_num = N_gaussian
            idx_cur = None

        position_features = self.GridEncoder(cur_xyz)
        position_features = F.normalize(position_features, p=2, dim=1)

        extracted_features = self.GaussianFeatureExtractor(cur_fea)
        extracted_features = position_features + extracted_features

        ctx_params_motion = extracted_features
        ctx_params_motion = ctx_params_motion[idx_cur].squeeze(0) if idx_cur is not None else ctx_params_motion

        z_hat_motion = decoder_factorized_chunk(self.EntropyFactorizedMotion._logits_cumulative, self.Q_z, sampled_num, self.lat_dim, z_hat_bit_path, chunk_size=10000)
        params_motion = self.MotionPriorDecoder(z_hat_motion)

        distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1))
        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6

        y_hat_motion = decoder_gaussian_chunk(mean_motion.contiguous(), std_motion, self.Q_y, y_hat_bit_path, chunk_size=100000)
        y_hat_motion = y_hat_motion.view(sampled_num, -1)

        dec_motion = self.MotionDecoder(y_hat_motion)
        nxt_gaussians = self.MotionCompensation(dynamicGS_type, dec_motion, cur_gaussians, idx_cur=idx_cur)

        # Refinement
        residual_ctx_params = extracted_features
        residual_feature = self.ResidualGenerator(residual_ctx_params)
        nxt_gaussians = self.Refinement(nxt_gaussians, residual_feature=residual_feature)

        # print('y_hat_motion: ', y_hat_motion, y_hat_motion.sum())
        # print('dec_motion: ', dec_motion)

        motion_xyz = nxt_gaussians._xyz - cur_xyz
        plot_hist(motion_xyz, title='motion_xyz', range=(-0.3, 0.3), bins=100, save_path=y_hat_bit_path.replace('.b', '_motion_xyz.png'))
        # print('motion_xyz: ', motion_xyz)  
        # print('motion_xyz: ', motion_xyz.max(), motion_xyz.min())
        # print('motion_xyz: ', motion_xyz.mean(), motion_xyz.std())
        # print('motion_xyz: ', motion_xyz.sum())

        return nxt_gaussians

    def forward(self):
        
        cur_xyz, cur_fea, N_gaussian = GaussianParameterPack(self.cur_gaussians)

        position_features = self.GridEncoder(cur_xyz)
        position_features = F.normalize(position_features, p=2, dim=1)
        extracted_features = self.GaussianFeatureExtractor(cur_fea)
        extracted_features = position_features + extracted_features

        # Spatial-temporal Context Prior
        ctx_params_motion = extracted_features # [N, latent_dim]

        # Sparse Motion Extration
        est_motion, idx_cur = self.MotionEstimation(self.args.dynamicGS_type, self.cur_gaussians, self.nxt_gaussians) # est_motion: [N_c, motion_dim]

        # Motion Encoding
        y_motion = self.MotionEncoder(est_motion) # [N_c, motion_dim]
        y_hat_motion = self.quantize(y_motion, Q=self.Q_y, train_flag=True)

        # Hyper Prior
        z_motion = self.MotionPriorEncoder(y_motion) # [N_c, latent_dim]
        z_hat_motion = self.quantize(z_motion, Q=self.Q_z, train_flag=True)
        params_motion = self.MotionPriorDecoder(z_hat_motion) # [N_c, latent_dim]

        # Motion Decoding
        dec_motion = self.MotionDecoder(y_hat_motion) # [N_c, motion_dim]

        # Motion Compensation
        nxt_gaussians = self.MotionCompensation(self.args.dynamicGS_type, dec_motion, self.cur_gaussians, idx_cur=idx_cur)
        with torch.no_grad():
            self.buffer = nxt_gaussians

        # Refinement
        residual_ctx_params = extracted_features
        residual_feature = self.ResidualGenerator(residual_ctx_params)
        nxt_gaussians = self.Refinement(nxt_gaussians, residual_feature=residual_feature)

        # Rate Estimation
        ctx_params_motion = ctx_params_motion[idx_cur].squeeze(0) if idx_cur is not None else ctx_params_motion
        distribution_motion = self.EntropyParametersMotion(torch.cat((ctx_params_motion, params_motion), dim=1)) if not self.without_context else self.EntropyParametersMotion_NO_CONTEXT(params_motion)

        mean_motion, std_motion = torch.chunk(distribution_motion, 2, dim=1)
        std_motion = F.softplus(std_motion) + 1e-6
        mean_motion = mean_motion.contiguous()

        bits_motion = self.EntropyGaussianMotion(y_hat_motion, mean_motion, std_motion, self.Q_y)
        bits_prior_motion = self.EntropyFactorizedMotion(z_hat_motion)
        total_size = bits_motion.mean() + bits_prior_motion.mean()  

        # Render Loss
        loss_render = self.ComputeRenderLoss(nxt_gaussians)

        return loss_render, total_size
    
    def test(self):
        for i in range(len(self.scene.getTestCameras().copy())):
            loss = self.ComputeRenderLoss(self.nxt_gaussians, False)
            print(f'loss_{i}: ', loss)