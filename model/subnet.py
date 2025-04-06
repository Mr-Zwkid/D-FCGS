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
    

class InceptionResNet(nn.Module):
    """
    Basic block: Inception Residual.
    """
    def __init__(self, channels):
        super(InceptionResNet, self).__init__()
        # path_0
        self.conv0_0 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True)
        self.conv0_1 = nn.Conv1d(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)

        # path_1
        self.conv1_0 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels // 4,
            kernel_size=1,
            stride=1,
            bias=True)
        self.conv1_1 = nn.Conv1d(
            in_channels=channels // 4,
            out_channels=channels // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        self.conv1_2 = nn.Conv1d(
            in_channels=channels // 4,
            out_channels=channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True)
        
        self.conv_trans = nn.Conv1d(
            in_channels=channels // 2 * 2,
            out_channels=channels,
            kernel_size=1,
            stride=1
        )
        

        self.relu = nn.ReLU()

    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = torch.cat([out0, out1], dim=1)
        out = self.conv_trans(out) + x if out.size(1) != x.size(1) else out + x

        return out

class ResNet(nn.Module):
    """
    Basic block: Residual
    """

    def __init__(self, channels):
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out
    
class DownsampleLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, resnet=InceptionResNet):
        super(DownsampleLayer, self).__init__()
        self.resnet = resnet
        self.conv = nn.Conv1d(input, hidden, kernel_size=1, stride=1, bias=True)
        self.down = nn.Conv1d(hidden, output, kernel_size=1, stride=1, bias=True)
        if resnet is not None:
            self.block = self.make_layer(resnet, block_layers, output)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(hidden)

        self.subnet = nn.Sequential(
            self.conv,
            self.norm,
            self.relu,
            self.down
        )

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.subnet(x)
        if self.resnet is not None:
            out = self.block(self.relu(out))

        return out
    

class ClipLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(ClipLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)
