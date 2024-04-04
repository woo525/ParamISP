from typing import Callable
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import kornia.filters as KF

from . import base
from .base import Conv2d, Linear, ChAttn2d, SpAttn2d


_DEBUG_IMAGES_: list[torch.Tensor] | None = None


def safe_pow(x: torch.Tensor, gamma: torch.Tensor | float, eps: float = 1e-12) -> torch.Tensor:
    """ Apply pow function to posibly negative values. """
    return x.sign() * (x.abs() + eps).pow(gamma)


def soft_histogram(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """ Compute soft histogram of each pixels.

    References:
        - Liu, Y.-L., et al. (2020). "Single-image HDR reconstruction by learning to reverse the camera pipeline." CVPR.

    Args:
        x: input tensor, `(*, H, W)`.
        n_bins: number of bins.

    Returns:
        output tensor, `(*, n_bins, H, W)`.
    """
    assert x.ndim >= 2, f"Expected image, got {x.shape}-dim"

    hists = torch.stack([
        F.relu(1. - torch.abs(x - i/(n_bins-1.)) * (n_bins-1.))
        for i in range(n_bins)
    ], dim=-3)
    assert hists.shape == (*x.shape[:-2], n_bins, *x.shape[-2:])

    return hists


def over_exposed_mask(x: torch.Tensor, threshold: float = 0.95) -> torch.Tensor:
    """ Compute soft over-exposed mask.

    Args:
        x: input tensor, `(*, H, W)`.
        threshold: threshold value.

    Returns:
        output tensor, `(*, H, W)`.
    """
    assert x.ndim >= 2, f"Expected image, got {x.shape}-dim"

    return (x - threshold).clip(min=0.) / (1.-threshold)


def coordinates(x: torch.Tensor, full_size: list[tuple[int, int]], top_left: list[tuple[int, int]], relative: bool = False) -> torch.Tensor:
    """ Compute coordinate mapping of each pixels.

    Args:
        x: input tensor, `(N, C, H, W)`.
        full_size: full size of each image, `[(H, W), ...]`.
        top_left: top-left position of each image, `[(y, x), ...]`.
        relative: whether to use relative coordinates.

    Returns:
        output tensor, `(N, 3, H, W)`.
    """
    
    assert x.ndim == 4, f"Expected batch image, got {x.shape}-dim"
    assert x.shape[0] == len(full_size) == len(top_left), \
        f"Expected same number of images, got {x.shape[0]}-batch, {len(full_size)}-full_size, {len(top_left)}-top_left"

    N, _, H, W = x.shape
    t = torch.tensor([top_left[b][0] for b in range(N)], dtype=x.dtype, device=x.device)   # (N,)
    l = torch.tensor([top_left[b][1] for b in range(N)], dtype=x.dtype, device=x.device)   # (N,)
    h = torch.tensor([full_size[b][0] for b in range(N)], dtype=x.dtype, device=x.device)  # (N,)
    w = torch.tensor([full_size[b][1] for b in range(N)], dtype=x.dtype, device=x.device)  # (N,)

    iy, ix = torch.meshgrid(torch.arange(H, dtype=x.dtype, device=x.device),
                            torch.arange(W, dtype=x.dtype, device=x.device), indexing="ij")  # (H, W)

    y = iy + (t - (h-1)/2.)[:, None, None]  # (N, H, W)
    x = ix + (l - (w-1)/2.)[:, None, None]  # (N, H, W)
    r = torch.sqrt(y**2 + x**2)             # (N, H, W)

    if relative:
        return torch.stack([
            y / h[:, None, None],
            x / w[:, None, None],
            r*2 / torch.sqrt((h-1)**2 + (w-1)**2)[:, None, None],
        ], dim=-3)  # (N, 3, H, W)
    else:
        return torch.stack([y, x, r], dim=-3)   # (N, 3, H, W)


def content_features(
    x: torch.Tensor,
    use_gradient: bool = True,
    histogram_bins: list[int] = [4, 8, 16],
):
    """ Create content features.

    Args:
        x: input tensor, `(N, C, H, W)`.
        use_gradient: whether to use gradient features.
        histogram_bins: list of of histogram bins for each channel.

    Returns:
        output tensor, `(N, C', H, W)`. `C' = C * (2 * use_gradient + sum(histogram_bins))`.
    """
    assert x.ndim >= 3, f"Expected batch image, got {x.shape}-dim"
    features: list[torch.Tensor] = []

    if use_gradient:
        features.append(KF.spatial_gradient(x).flatten(-4, -3))  # (N, C*2, H, W)

    for n_bins in histogram_bins:
        features.append(soft_histogram(x, n_bins).flatten(-4, -3))  # (N, C*n_bins, H, W)

    return torch.cat(features, dim=-3)  # (N, C', H, W)


def coordinate_features(
    x: torch.Tensor,
    full_size: list[tuple[int, int]],
    top_left:  list[tuple[int, int]],
    use_relative:    bool  = True,
    use_absolute:    bool  = True,
    use_frequency:   bool  = True,
    absolute_scale:  float = 3000.,
    frequency_scale: float = 4.,
    frequency_bits:  int   = 6,
) -> torch.Tensor:
    """ Create coordinate features with positional embedding.

    Args:
        x: input tensor, `(N, C, H, W)`.
        full_size: full size of each image, `[(H, W), ...]`.
        top_left: top-left position of each image, `[(y, x), ...]`.
        use_relative: whether to use relative coordinates.
        absolute_scale: dividing factor for normalizing absolute coordinates.
        frequency_scale: dividing factor for frequency domains.
        frequency_bits: number of bits for frequency domains.

    Returns:
        output tensor, `(N, C', H, W)`. :math:`C' = 3 * (use_relative + use_absolute + use_frequency * frequency_bits * 2)`.
    """
    abscoords = coordinates(x, full_size, top_left, relative=False)  # (N, 3, H, W)
    features: list[torch.Tensor] = []

    if use_relative:
        features.append(coordinates(x, full_size, top_left, relative=True))  # (N, 3, H, W)

    if use_absolute:
        features.append(abscoords / absolute_scale)

    if use_frequency:
        for i in range(frequency_bits):
            features.append(torch.sin(abscoords / (frequency_scale**i)))
            features.append(torch.cos(abscoords / (frequency_scale**i)))

    """ exp section
    relcoords = coordinates(x, full_size, top_left, relative=True)  # (N, 3, H, W)
    features: list[torch.Tensor] = []

    if use_relative:
        features.append(relcoords)  # (N, 3, H, W)

    if use_frequency:
        for i in range(frequency_bits):
            features.append(torch.sin(relcoords * (2.**i)))
            features.append(torch.cos(relcoords * (2.**i)))
    """

    return torch.cat(features, dim=-3)  # (N, C', H, W)


class GlobalNet(nn.Module):
    """ Global tone mapping network.

    Args:
        feat_channels: number of channels in input feature.
        mid_channels: number of channels in middle layers.
        num_conv: number of convolution layers for global feature extraction.
        num_fc: number of fully-connected layers for output embedding.
        num_stage: number of stages applying tone mapping.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        z: input feature, `(*, feat_channels)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """
    color_channels: int = 3

    def __init__(
        self,
        feat_channels: int,
        mid_channels: int = 64,
        num_conv: int = 3,
        num_fc: int = 3,
        num_stage: int = 4,
        **kwargs,
    ):
        super().__init__()
        assert num_conv >= 1, f"`num_conv` must be positive, got {num_conv}"
        assert num_fc >= 1, f"`num_fc` must be positive, got {num_fc}"
        assert num_stage >= 1, f"`num_stage` must be positive, got {num_stage}"

        self.feat_channels = feat_channels
        self.mid_channels = mid_channels
        self.num_conv = num_conv
        self.num_fc = num_fc
        self.num_stage = num_stage
        
        self.convs = nn.ModuleList(
            [Conv2d(feat_channels, mid_channels, 3, stride=2, padding=0, **kwargs)] +
            [Conv2d(mid_channels,  mid_channels, 3, stride=2, padding=0, **kwargs) for _ in range(num_conv-1)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fcs = nn.ModuleList(
            [Linear(mid_channels, mid_channels, **kwargs) for _ in range(num_fc-1)]
        )
        self.fc_index = nn.Linear(num_stage, mid_channels)
        self.fc_last = nn.ModuleList(
            [Linear(mid_channels, 13*self.color_channels, **kwargs) for _ in range(num_stage)]
        )

        self.act = nn.ELU(inplace=True)

        if "init_weights" not in kwargs:
            for fc in self.fc_last:  # type: ignore
                fc: Linear
                init.xavier_normal_(fc.weight, 0.01)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3], f"Expected same batch size, got {x.shape[:-3]} and {z.shape[:-3]}"
        assert x.shape[-3] == self.color_channels, f"Expected {self.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.feat_channels, f"Expected {self.feat_channels}-channel feature, got {z.shape[-3]}"

        z = self.extract_global_features(z)  # (*, mid_channels)

        y = x
        for stage in torch.arange(self.num_stage, device=x.device):
            w = self.embed_features(z, stage)  # (*, 10*3+3*3)
            y = self.apply_tone_mapping(y, w)

        return y

    def extract_global_features(self, z: torch.Tensor) -> torch.Tensor:
        """ Extract global features from the spatial features.

        Args:
            feat: spatial features, `(*, feat_channels, H, W)`.

        Returns:
            global features, `(*, mid_channels, H/2^num_conv, W/2^num_conv)`.
        """
        assert z.ndim >= 3, f"Expected spatial feature, got {z.shape}-dim"
        assert z.shape[-3] == self.feat_channels, f"Expected {self.feat_channels} channels, got {z.shape[-3]}"

        for conv in self.convs:
            z = self.pool(self.act(conv(z)))    # (*, num_channels, H/2^i, W/2^i)

        return z.mean(dim=(-2, -1))             # (*, num_channels)

    def embed_features(self, w: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """ Embed features with layer index.

        Args:
            w: global features, `(*, mid_channels)`.
            i: layer index, must be scalar.

        Returns:
            embedded weights, `(*, mid_channels)`.
        """
        assert w.ndim >= 1, f"Expected global feature, got {w.shape}-dim"
        assert w.shape[-1] == self.mid_channels, f"Expected {self.mid_channels} channels, got {w.shape[-1]}"

        i_sparse = F.one_hot(i, self.num_stage).float()  # (*, num_stage)
        w = w + self.act(self.fc_index(i_sparse))

        for fc in self.fcs:
            w = self.act(fc(w))

        w = self.act(self.fc_last[i](w))

        return w

    @classmethod
    def apply_tone_mapping(cls, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """ Apply tone mapping to the input tensor.

        Args:
            x: input tensor, `(*, 3, H, W)`.
            w: tone mapping coefficients, `(*, 10*3+3*3)`.

        Returns:
            output tensor, `(*, 3, H, W)`.
        """
        assert x.ndim == 4, f"Expected 4-dim tensor, got {x.shape}-dim"
        assert w.ndim == 2, f"Expected 2-dim tensor, got {w.shape}-dim"
        assert x.shape[-3] == cls.color_channels, f"Expected {cls.color_channels} channels, got {x.shape[-3]}"

        w_q = w[..., :10*cls.color_channels].reshape(*w.shape[:-1], 10, 3)
        w_g = w[..., 10*cls.color_channels:].reshape(*w.shape[:-1], 3, 3)

        y = x
        y = y.flatten(-2).permute(2, 0, 1)  # (H*W, N, C)

        y = y + cls._apply_quadratic_function(y, w_q)
        if _DEBUG_IMAGES_ is not None:
            _DEBUG_IMAGES_.append(y.permute(1, 2, 0).reshape(*x.shape))

        y = cls._apply_extended_gamma_curve(y, w_g)
        if _DEBUG_IMAGES_ is not None:
            _DEBUG_IMAGES_.append(y.permute(1, 2, 0).reshape(*x.shape))

        y = y.permute(1, 2, 0).reshape(*x.shape)  # (N, C, H, W)
        return y

    @classmethod
    def _apply_quadratic_function(cls, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """ Apply quadratic function maps.

        Args:
            x: input tensor, `(H*W, N, C)`.
            w: Quadratic coefficients, `(*, (C+1)*(C+2)/2, C)`.

        Returns:
            output tensor, `(H*W, N, C)`.
        """
        C = x.shape[-1]
        assert w.shape[-2:] == (((C+1)*(C+2))//2, C), \
            f"Expected quadratic coefficients with shape (*, {(C+1)*(C+2)//2}, {C}), got {w.shape}"

        y = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)  # (H*W, N, C+1)

        y = torch.matmul(y.unsqueeze(-1), y.unsqueeze(-2))       # (H*W, N, C+1, C+1)
        assert y.shape[-2:] == (C+1, C+1)

        triu_indices = torch.triu_indices(C+1, C+1)              # (2, (C+1)*(C+2)/2)
        y = y.triu()[..., triu_indices[0], triu_indices[1]]      # (H*W, N, (C+1)*(C+2)/2)
        assert y.shape[-1] == ((C+1)*(C+2))//2

        y = y.unsqueeze(-2)                                      # (H*W, N, 1,(C+1)*(C+2)/2)
        y = torch.matmul(y, w)                                   # (H*W, N, 1, C)
        y = y.squeeze(-2)                                        # (H*W, N, C)
        assert y.shape[-1] == C

        return y

    @classmethod
    def _apply_extended_gamma_curve(cls, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        r""" Apply extended Gamma function.

        Let :math:`s = t^\gamma` where :math:`t = t_1 x + t_0, s = s_1 y + s_0`,
        :math:`s_0 = t_0^\gamma, (s_1 + s_0) = (t_1 + t_0)^\gamma` and :math:`s_1, s_0, \gamma > 0`,
        then :math:`y = f(x)` is monotonically increasing function and :math:`f(0) = 0, f(1) = 1`
        when :math:`x \in [0, 1]`. The parameters :math:`t_0, t_1, \gamma` are learnable.

        Args:
            x: input tensor, (H*W, N, C).
            w: :math:`log(\gamma), t_0, log(t_1)` parameters, (*, 3, C).

        Returns:
            output tensor with shape (H*W, N, C).
        """
        assert x.shape[-1] == w.shape[-1], f"Expected same number of channels, got {x.shape[-1]} and {w.shape[-1]}"

        gamma = w[..., 0, :].exp()
        t0    = w[..., 1, :].abs()
        t1    = w[..., 2, :].exp()
        s0 = safe_pow(t0, gamma)
        s1 = safe_pow(t1+t0, gamma) - s0

        t = t1 * x + t0
        s = safe_pow(t, gamma)
        y = (s - s0) / s1

        return y


class LocalBlock(nn.Module):
    """ Residual convolution-attention block.

    Reference:
        - Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." ECCV.

    Args:
        n_channels: number of feature channels.
        reduction: scale of feature channels reduction: `hidden_channels = n_channels // reduction`.
        kernel_size: size of the convolution kernel.
    """

    def __init__(self, n_channels: int, reduction: int = 2, *, kernel_size: int = 3, **kwargs):
        super().__init__()
        assert kernel_size % 2 == 1, f"`kernel_size` must be odd, got {kernel_size}"
        self.conv1 = Conv2d(n_channels, n_channels//reduction, kernel_size, padding_mode="replicate", **kwargs)
        self.conv2 = Conv2d(n_channels//reduction, n_channels, kernel_size, padding_mode="replicate", **kwargs)
        self.chattn = ChAttn2d(n_channels, reduction, **kwargs)
        self.spattn = SpAttn2d(padding_mode="replicate", **kwargs)
        self.act = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        r = self.conv1(r)
        r = self.act(r)
        r = self.conv2(r)
        r = self.chattn(r)
        r = self.spattn(r)
        return x + r


class LocalNet(nn.Module):
    """ Local residual network with U-Net style structure.

    Args:
        feat_channels: number of channels in input feature.
        mid_channels: number of channels in middle layers.
        reduction: scale of feature channels reduction: `hidden_channels = n_channels // reduction`.
        kernel_size: size of the convolution kernel.
        num_block: number of residual blocks in each stage.
        num_scale: number of scales in U-Net structure, 0 for no down/up-sampling.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        z: input feature, `(*, feat_channels)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """
    color_channels: int = 3

    def __init__(
        self,
        feat_channels: int,
        mid_channels: int = 64,
        reduction: int = 2,
        kernel_size: int = 3,
        num_block: int = 2,
        num_scale: int = 2,
        **kwargs,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, f"`kernel_size` must be odd, got {kernel_size}"
        assert num_block >= 1, f"`num_block` must be positive, got {num_block}"
        assert num_scale >= 0, f"`num_scale` must be non-negative, got {num_scale}"

        self.feat_channels = feat_channels
        self.mid_channels = mid_channels
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.num_block = num_block
        self.num_scale = num_scale

        self.conv_head = Conv2d(feat_channels, mid_channels, kernel_size, padding_mode="replicate", **kwargs)
        self.conv_down = nn.ModuleList([
            nn.Conv2d(mid_channels, mid_channels, 2, stride=2)
            for _ in range(num_scale)
        ])
        self.conv_up = nn.ModuleList([
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2)
            for _ in range(num_scale)
        ])
        self.conv_tail = Conv2d(mid_channels, self.color_channels, kernel_size, padding_mode="replicate", **kwargs)

        self.resblks = nn.ModuleList([
            LocalBlock(mid_channels, reduction, kernel_size=kernel_size, **kwargs)
            for _ in range(num_block * (2*num_scale + 1))
        ])

        if "init_weights" not in kwargs:
            init.xavier_normal_(self.conv_tail.weight, 1e-2)
            if self.conv_tail.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv_tail.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(self.conv_tail.bias, -bound, bound)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3], f"Expected same batch size, got {x.shape[:-3]} and {z.shape[:-3]}"
        assert x.shape[-3] == self.color_channels, f"Expected {self.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.feat_channels, f"Expected {self.feat_channels}-channel feature, got {z.shape[-3]}"
        assert x.shape[-1] % 2**self.num_scale == 0 and x.shape[-2] % 2**self.num_scale == 0, \
            f"Expected image size divisible by 2^{self.num_scale}, got {x.shape[-2:]}"

        z_skip: dict[int, torch.Tensor] = {}

        z = self.conv_head(z)  # (*, mid_channels, H, W)

        for stage in range(self.num_scale):
            for i in range(stage * self.num_block, (stage + 1) * self.num_block):
                z = self.resblks[i](z)
            z_skip[stage] = z
            z = self.conv_down[stage](z)  # (*, mid_channels, H/2, W/2)

        for i in range(self.num_scale * self.num_block, (self.num_scale + 1) * self.num_block):
            z = self.resblks[i](z)

        for stage in range(self.num_scale-1, -1, -1):
            z = self.conv_up[stage](z)
            z = z_skip[stage] + z
            for i in range((self.num_scale+stage+1) * self.num_block, (self.num_scale+stage+2) * self.num_block):
                z = self.resblks[i](z)

        z = self.conv_tail(z)  # (*, 3, H, W)

        return x + z


class HyperGlobalNet(nn.Module):
    """ Hypernetwork for `GlobalNet`.

    Args:
        target: target network.
        n_params: dimension of hyperparameters.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        z: input feature, `(*, feat_channels)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, target: GlobalNet, n_params: int, **kwargs):
        super().__init__()
        self.target = target
        self.n_params = n_params

        self.fc_embed = Linear(n_params, target.mid_channels, **kwargs)
        if "init_weights" not in kwargs:
            init.xavier_normal_(self.fc_embed.weight, 0.1)

    def forward(self, x: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert h.ndim >= 1, f"`h` must be hyperparameters, got {h.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3] == h.shape[:-1], \
            f"Expected same batch size, got {x.shape[:-3]}, {z.shape[:-3]} and {h.shape[:-1]}"
        assert x.shape[-3] == self.target.color_channels, \
            f"Expected {self.target.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.target.feat_channels, \
            f"Expected {self.target.feat_channels}-channel feature, got {z.shape[-3]}"

        # print(1, x.isnan().sum().item(), z.isnan().sum().item(), h.isnan().sum().item())
        z = self.target.extract_global_features(z)  # (*, mid_channels)
        z = z + self.target.act(self.fc_embed(h))

        y = x
        for stage in torch.arange(self.target.num_stage, device=x.device):
            w = self.target.embed_features(z, stage)  # (*, 10*3+3*3)
            y = self.target.apply_tone_mapping(y, w)

        return y


class RealHyperGlobalNet(nn.Module):
    """ Hypernetwork for `GlobalNet`.

    Args:
        target: target network.
        n_params: dimension of hyperparameters.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        z: input feature, `(*, feat_channels)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, target: GlobalNet, n_params: int, **kwargs):
        super().__init__()
        self.target = target
        self.n_params = n_params

        self.hfcs = nn.ModuleList(base.HyperLinear.build(n_params, target.fcs, **kwargs))  # type: ignore

    def forward(self, x: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert h.ndim >= 1, f"`h` must be hyperparameters, got {h.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3] == h.shape[:-1], \
            f"Expected same batch size, got {x.shape[:-3]}, {z.shape[:-3]} and {h.shape[:-1]}"
        assert x.shape[-3] == self.target.color_channels, \
            f"Expected {self.target.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.target.feat_channels, \
            f"Expected {self.target.feat_channels}-channel feature, got {z.shape[-3]}"

        z = self.target.extract_global_features(z)  # (*, mid_channels)

        y = x
        for stage in torch.arange(self.target.num_stage, device=x.device):
            w = self.embed_features(z, stage, h)  # (*, 10*3+3*3)
            y = self.target.apply_tone_mapping(y, w)

        return y

    def embed_features(self, w: torch.Tensor, i: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ Embed features with layer index.

        Args:
            w: global features, `(*, mid_channels)`.
            i: layer index, must be scalar.

        Returns:
            embedded weights, `(*, mid_channels)`.
        """
        assert w.ndim >= 1, f"Expected global feature, got {w.shape}-dim"
        assert w.shape[-1] == self.target.mid_channels, f"Expected {self.target.mid_channels} channels, got {w.shape[-1]}"

        i_sparse = F.one_hot(i, self.target.num_stage).float()  # (*, num_stage)
        w = w + self.target.act(self.target.fc_index(i_sparse))

        for fc in self.hfcs:
            w = self.target.act(fc(w, h))

        w = self.target.act(self.target.fc_last[i](w))

        return w


class HyperLocalBlock(nn.Module):
    """ Residual convolution-attention block with parameter modulation.

    Args:
        target: The target ResBlock to be modulated.
        hconv1: HyperConv2d for the 1st convolution of the target.
        hconv2: HyperConv2d for the 2nd convolution of the target.
        hchattn: HyperChAttn2d for the channel attention of the target.
        hspattn: HyperSpAttn2d for the spatial attention of the target.

    Inputs:
        x: input tensor, `(*, n_channels, H, W)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, n_channels, H, W)`.
    """

    def __init__(self, target: LocalBlock,
                 hconv1: base.HyperConv2d, hconv2: base.HyperConv2d,
                 hchattn: base.HyperChAttn2d, hspattn: base.HyperSpAttn2d):
        super().__init__()
        self.target = target
        self.hconv1 = hconv1
        self.hconv2 = hconv2
        self.hchattn = hchattn
        self.hspattn = hspattn

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        r = x
        r = self.hconv1(r, h)
        r = self.target.act(r)
        r = self.hconv2(r, h)
        r = self.hchattn(r, h)
        r = self.hspattn(r, h)
        return x + r

    @classmethod
    def build(cls, n_params: int, targets: list[LocalBlock],
              mid_channels: int = 64, init_weights: Callable = lambda x: init.xavier_normal_(x, 2e-3)):
        """ Build a list of hyper-modules for each target modules.

        Args:
            n_params: The number of hyperparameters.
            targets: The list of ResBlock.
            init_weights: The weight initialization function.

        Returns:
            A list of HyperResBlock.
        """
        n_layers = len(targets)
        hconv1s = base.HyperConv2d.build(n_params, [t.conv1 for t in targets],
                                         mid_channels=mid_channels, init_weights=init_weights)
        hconv2s = base.HyperConv2d.build(n_params, [t.conv2 for t in targets],
                                         mid_channels=mid_channels, init_weights=init_weights)
        hchattns = base.HyperChAttn2d.build(n_params, [t.chattn for t in targets], init_weights=init_weights)
        hspattns = base.HyperSpAttn2d.build(n_params, [t.spattn for t in targets], init_weights=init_weights)

        return [cls(targets[i], hconv1s[i], hconv2s[i], hchattns[i], hspattns[i]) for i in range(n_layers)]


class HyperLocalNet(nn.Module):
    """ Hypernetwork for `LocalNet`.

    Args:
        target: target network.
        n_params: dimension of hyperparameters.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        z: input feature, `(*, feat_channels, H, W)`.
        h: hyperparameters, `(*, n_params)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, target: LocalNet, n_params: int, **kwargs):
        super().__init__()
        self.target = target
        self.n_params = n_params

        self.hresblks = nn.ModuleList(HyperLocalBlock.build(n_params, target.resblks, **kwargs))  # type: ignore

    def forward(self, x: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert h.ndim >= 1, f"`h` must be hyperparameters, got {h.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3] == h.shape[:-1], \
            f"Expected same batch size, got {x.shape[:-3]}, {z.shape[:-3]} and {h.shape[:-1]}"
        assert x.shape[-3] == self.target.color_channels, \
            f"Expected {self.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.target.feat_channels, \
            f"Expected {self.feat_channels}-channel feature, got {z.shape[-3]}"

        z_skip: dict[int, torch.Tensor] = {}

        z = self.target.conv_head(z)

        for stage in range(self.target.num_scale):
            for i in range(stage * self.target.num_block, (stage + 1) * self.target.num_block):
                z = self.hresblks[i](z, h)
            z_skip[stage] = z
            z = self.target.conv_down[stage](z)

        for i in range(self.target.num_scale * self.target.num_block, (self.target.num_scale + 1) * self.target.num_block):
            z = self.hresblks[i](z, h)

        for stage in range(self.target.num_scale-1, -1, -1):
            z = self.target.conv_up[stage](z)
            z = z_skip[stage] + z
            for i in range((self.target.num_scale+stage+1) * self.target.num_block, (self.target.num_scale+stage+2) * self.target.num_block):
                z = self.hresblks[i](z, h)

        z = self.target.conv_tail(z)

        return x + z


class SynHyperLocalNet(nn.Module):
    def __init__(self, target: LocalNet, n_params: int, **kwargs):
        super().__init__()
        self.target = target
        self.n_params = n_params

        self.fc_head = Linear(n_params, target.mid_channels, **kwargs)

    def forward(self, x: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"`x` must be images, got {x.shape}-dim"
        assert z.ndim >= 3, f"`z` must be spatial features, got {z.shape}-dim"
        assert h.ndim >= 1, f"`h` must be hyperparameters, got {h.shape}-dim"
        assert x.shape[-2:] == z.shape[-2:], f"Expected same image size, got {x.shape[-2:]} and {z.shape[-2:]}"
        assert x.shape[:-3] == z.shape[:-3] == h.shape[:-1], \
            f"Expected same batch size, got {x.shape[:-3]}, {z.shape[:-3]} and {h.shape[:-1]}"
        assert x.shape[-3] == self.target.color_channels, \
            f"Expected {self.target.color_channels}-channel image, got {x.shape[-3]}"
        assert z.shape[-3] == self.target.feat_channels, \
            f"Expected {self.target.feat_channels}-channel feature, got {z.shape[-3]}"

        z_skip: dict[int, torch.Tensor] = {}

        z = self.target.conv_head(z)  # (*, mid_channels, H, W)
        h = self.fc_head(h)  # (*, mid_channels)
        z = z + h.view(h.shape + (1, 1))  # (*, mid_channels, H, W)

        for stage in range(self.target.num_scale):
            for i in range(stage * self.target.num_block, (stage + 1) * self.target.num_block):
                z = self.target.resblks[i](z)
            z_skip[stage] = z
            z = self.target.conv_down[stage](z)  # (*, mid_channels, H/2, W/2)

        for i in range(self.target.num_scale * self.target.num_block, (self.target.num_scale + 1) * self.target.num_block):
            z = self.target.resblks[i](z)

        for stage in range(self.target.num_scale-1, -1, -1):
            z = self.target.conv_up[stage](z)
            z = z_skip[stage] + z
            for i in range((self.target.num_scale+stage+1) * self.target.num_block, (self.target.num_scale+stage+2) * self.target.num_block):
                z = self.target.resblks[i](z)

        z = self.target.conv_tail(z)  # (*, 3, H, W)

        return x + z
