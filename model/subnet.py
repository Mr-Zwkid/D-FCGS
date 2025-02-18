import torch
import torch.nn as nn

class GDN1D(nn.Module):
    def __init__(self, num_features, inverse=False, beta_min=1e-6, gamma_init=0.1):
        super(GDN1D, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        # 可学习的参数
        self.beta = nn.Parameter(torch.ones(num_features))
        self.gamma = nn.Parameter(torch.eye(num_features) * gamma_init)

    def forward(self, x):
        # 确保 beta 不小于 beta_min
        beta = torch.max(self.beta, torch.tensor(self.beta_min, dtype=torch.float32, device=x.device))
        # 计算归一化因子
        norm_pool = torch.einsum('bi,ij->bj', x ** 2, self.gamma) + beta
        norm_pool = torch.sqrt(norm_pool)

        if self.inverse:
            output = x * norm_pool
        else:
            output = x / norm_pool

        return output

class MaskedConv1d(nn.Conv1d):
    r"""Masked 1D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Inherits the same arguments as a `nn.Conv1d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the
    following layers.
    """

    def __init__(self, *args, mask_type="A", **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ("A", "B"):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        # 初始化掩码
        self.register_buffer("mask", torch.ones_like(self.weight.data))
        _, _, w = self.mask.size()

        # 根据掩码类型设置掩码
        center = w // 2
        if mask_type == "A":
            self.mask[:, :, center:] = 0
        else:
            self.mask[:, :, center + 1:] = 0

    def forward(self, x):
        # 应用掩码到权重上
        self.weight.data *= self.mask
        return super().forward(x)