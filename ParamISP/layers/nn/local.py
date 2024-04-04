from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .base import Conv2d, Linear, ChAttn2d, SpAttn2d


class ResAttnBlock(nn.Module):
    """ ResBlock integrated with Convolutional Block Attention Module (CBAM).

    Reference:
        - Woo, S., et al. (2018). "CBAM: Convolutional Block Attention Module." ECCV.

    Args:
        n_channels: number of feature channels.
        kernel_size: kernel size of head convolutional layers.
        reduction: scale of feature channels reduction: `hidden_channels = n_channels // reduction`.
        dilation: dilation of head convolutional layers.
        groups: number of groups for head convolutional layers.
        bias: whether to use bias in head convolutional layers.
        padding_mode: padding mode of head convolutional layers.
        init_weights: function to initialize weights.
        ch_attn_builder: function to build channel attention module.
        sp_attn_builder: function to build spatial attention module.
    """

    def __init__(self, n_channels: int, kernel_size: int = 3, reduction: int = 4, *,
                 ch_attn_builder: Callable[[int], ChAttn2d] = ChAttn2d,
                 sp_attn_builder: Callable[[],    SpAttn2d] = SpAttn2d,
                 dilation: int = 1, groups: int = 1, bias: bool = True, padding_mode: str = "zeros",
                 init_weights: Callable = init.xavier_normal_):
        super().__init__()
        self.conv1 = Conv2d(n_channels, n_channels//reduction, kernel_size, dilation=dilation,
                            groups=groups, bias=bias, padding_mode=padding_mode, init_weights=init_weights)
        self.conv2 = Conv2d(n_channels//reduction, n_channels, kernel_size, dilation=dilation,
                            groups=groups, bias=bias, padding_mode=padding_mode, init_weights=init_weights)
        self.act = F.elu
        self.ch_attn = ch_attn_builder(n_channels)
        self.sp_attn = sp_attn_builder()

    def head_conv(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.act(self.conv1(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.head_conv(x)
        r = self.ch_attn(r)
        r = self.sp_attn(r)
        return x + r


class LocalResBlock(nn.Module):
    """ Local residual block, inspired by Low Level stage of DeepISP.

    References:
        - Schwartz, E., et al. (2018). "DeepISP: Toward learning an end-to-end image processing pipeline." IEEE TIP 28(2): 912-923.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        color_channels: Number of feature channels; See Outputs section.
        **kwargs: Additional arguments passed to `Conv2d`.

    Inputs:
        x: input tensor of shape (batch_size, in_channels, height, width).
        The first `color_channels` channels are treated as color channels, and the rest are treated as feature channels.

    Outputs:
        output tensor of shape (batch_size, output_channels, height, width).
        The first `color_channels` channels are treated as color channels, and the rest are treated as feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int, color_channels: int = 3, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.color_channels = color_channels

        assert in_channels  >= color_channels, f"Invalid config `{in_channels=}` < `{color_channels=}`"
        assert out_channels >= color_channels, f"Invalid config `{out_channels=}` < `{color_channels=}`"

        self.conv = Conv2d(in_channels, out_channels, **kwargs)
        self.act_c = torch.tanh
        self.act_f = F.leaky_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"`x` must be 4-dim, got {x.ndim}-dim"
        assert x.size(1) == self.in_channels, \
            f"`x` must have {self.in_channels} channels, got {x.size(1)} channels"

        C = self.color_channels
        y = self.conv(x)
        y_c, y_f = y[:, :C], y[:, C:]
        y_c = self.act_c(y_c) + x[:, :C]
        y_f = self.act_f(y_f)
        y = torch.cat([y_c, y_f], dim=1)

        return y


class LocalNet(nn.Module):
    """ Local network, inspired by Low Level stage of DeepISP.

    References:
        - Schwartz, E., et al. (2018). "DeepISP: Toward learning an end-to-end image processing pipeline." IEEE TIP 28(2): 912-923.

    Args:
        num_blocks: Number of residual blocks.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        mid_channels: Number of hidden channels.
        color_channels: Number of feature channels; See Outputs section.
        kernel_size: Kernel size of convolution layers.
        dialated: Whether to use dialated convolution in even blocks.
        **kwargs: Additional arguments passed to `Conv2d`.

    Inputs:
        x: input tensor of shape (batch_size, in_channels, height, width).
        The first `color_channels` channels are treated as color channels, and the rest are treated as feature channels.

    Outputs:
        output tensor of shape (batch_size, out_channels, height, width).
        The first `color_channels` channels are treated as color channels, and the rest are treated as feature channels.
    """

    def __init__(self, num_blocks: int, in_channels: int, out_channels: int,
                 mid_channels: int = 64, color_channels: int = 3, kernel_size: int = 3, dialated: bool = True, **kwargs):
        super().__init__()

        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.color_channels = color_channels

        self.blocks = nn.ModuleList(
            [LocalResBlock(in_channels,  mid_channels, color_channels, kernel_size=kernel_size, **kwargs)] +
            [LocalResBlock(mid_channels, mid_channels, color_channels, kernel_size=kernel_size,
                           dilation=(2 if dialated and i % 2 == 1 else 1), **kwargs)
                for i in range(1, num_blocks-1)] +
            [LocalResBlock(mid_channels, out_channels, color_channels, kernel_size=kernel_size, **kwargs)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"`x` must be 4-dim, got {x.ndim}-dim"
        assert x.size(1) == self.in_channels, \
            f"`x` must have {self.in_channels} channels, got {x.size(1)} channels"

        y = x
        for block in self.blocks:
            y = block(y)

        return y


def gamma_correction(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """ Apply Gamma correction.

    Args:
        x: input tensor of shape (batch_size, color_channels, height, width).
        gamma: Gamma value.
        eps: Small value to avoid zero division; negative values are point-symmetrically mapped to positive values.

    Outputs:
        output tensor of shape (batch_size, color_channels, height, width).
    """
    assert x.ndim == 4, f"`x` must be 4-dim, got {x.ndim}-dim"
    assert x.size(1) == 3, f"`x` must have 3 channels, got {x.size(1)} channels"

    if gamma.nelement() == 1:
        gamma = gamma.flatten()
    elif gamma.nelement() == x.size(0):
        assert gamma.size(0) == x.size(0), f"`gamma` must have {x.size(0)} elements, got {gamma.size(0)} elements"
        gamma = gamma.flatten()
    else:
        assert gamma.ndim == 2, f"`gamma` must be 1-dim or 2-dim, got {gamma.ndim}-dim"
        assert gamma.size(0) == x.size(0), f"`gamma` must have {x.size(0)} elements, got {gamma.size(0)} elements"
        assert gamma.size(1) == x.size(1), f"`gamma` must have {x.size(1)} channels, got {gamma.size(1)} channels"

    y = x.sign() * (x.abs() + eps) ** gamma
    return y


class GammaCorrection(nn.Module):
    """ Learnable Gamma correction.

    Args:
        gamma: Initial value of gamma.

    Inputs:
        x: input tensor of shape (batch_size, color_channels, height, width).

    Outputs:
        output tensor of shape (batch_size, color_channels, height, width).
    """

    def __init__(self, gamma: float = 1/2.2, min: float = 1/4.0, max: float = 1.0, eps=1e-6):
        super().__init__()
        self.gamma = nn.parameter.Parameter(torch.tensor(gamma))
        self.min = min
        self.max = max
        self.eps = eps

    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        self.gamma.data.clamp_(self.min, self.max)
        gamma = self.gamma
        if inverse:
            gamma = 1 / gamma
        return gamma_correction(x, gamma, self.eps)
