from typing import Callable
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import kornia.filters as KF

import data.emor

from .base import Conv2d, Linear


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


def apply_quadratic_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """ Apply quadratic function to each pixels with given parameters.

    References:
        - Schwartz, E., et al. (2018). "DeepISP: Toward learning an end-to-end image processing pipeline." IEEE TIP 28(2): 912-923.

    Args:
        x: input tensor, `(*, 3, H, W)`.
        w: parameters tensor, `(*, 10, 3)`.

    Returns:
        output tensor, `(*, 3, H, W)`.
    """
    assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"
    BDIM = x.shape[:-3]
    C, H, W = x.shape[-3:]
    assert C == 3, f"`x` must have 3 channels, got {C} channels"
    assert w.shape[-2:] == (10, 3), f"`w` must have shape (*, 10, 3), got {w.shape}"

    y = x.flatten(start_dim=-2).swapaxes(-1, -2)            # (*, H*W, 3)
    assert y.shape == (*BDIM, H*W, 3)

    y = y.reshape(-1, y.size(-1))                           # (N*H*W, 3)
    y = torch.cat([y, torch.ones_like(y[:, :1])], dim=1)    # (N*H*W, 4)

    y = torch.bmm(y.unsqueeze(-1), y.unsqueeze(-2))         # (N*H*W, 4, 4)
    assert y.shape[-2:] == (4, 4)

    triu_indices = [0, 1, 2, 3, 5, 6, 7, 10, 11, 15]        # (R^2, R*G, R*B, R, G^2, G*B, G, B^2, B, 1)
    y = y.flatten(start_dim=-2)[..., triu_indices]          # (N*H*W, 10)
    y = y.reshape(*BDIM, H*W, -1)                           # (*, H*W, 10)
    assert y.shape == (*BDIM, H*W, 10)

    y = torch.matmul(y, w)                                  # (*, H*W, 3)
    assert y.shape[-2:] == (H*W, 3)

    y = y.swapaxes(-2, -1).reshape(*BDIM, C, H, W)          # (*, 3, H, W)
    return y


def apply_gamma_correction(x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """ Apply gamma correction to each pixels with given parameters.

    Args:
        x: input tensor, `(*, C, H, W)`.
        gamma: parameters tensor, `(*, 1)`.

    Returns:
        output tensor, `(*, C, H, W)`.
    """
    assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"
    BDIM = x.shape[:-3]
    assert gamma.shape[-1] == 1, f"`gamma` must have shape (*, 1), got {gamma.shape}"
    eps = 1e-12

    gamma = gamma.reshape(*BDIM, 1, 1, 1)       # (*, 1, 1, 1)
    y = x.sign() * (x.abs() + eps).pow(gamma)   # (*, C, H, W)
    return y


def apply_response_function(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """ Apply CRF to each pixels with given parameters.

    References:
        - Liu, Y.-L., et al. (2020). "Single-image HDR reconstruction by learning to reverse the camera pipeline." CVPR.

    Args:
        x: input tensor, `(*, C, H, W)`.
        w: parameters tensor, `(*, 1024)`.

    Returns:
        output tensor, `(*, C, H, W)`.
    """
    assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"
    BDIM = x.shape[:-3]
    C, H, W = x.shape[-3:]
    assert w.shape[-1] == 1024, f"`w` must have shape (*, 1024), got {w.shape}"

    y = x.flatten(start_dim=-3)                           # (*, C*H*W)
    y0 = (y.clip(0, 1) * 1023).floor().long()
    y1 = y0 + 1
    w0 = y1 - y
    w1 = y - y0
    y1 = y1.clip(0, 1023)

    # sample from w
    y0 = torch.gather(w, -1, y0)
    y1 = torch.gather(w, -1, y1)

    # interpolate
    y = w0 * y0 + w1 * y1

    y = y.reshape(*BDIM, C, H, W)                         # (*, C, H, W)
    return y


class DirectQuadraticNet(nn.Module):
    """ Directly learn quadratic tone mapping parameters.

    Args:
        num_levels: number of applying quadratic function.
        mid_channels: number of channels in the middle layers.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, num_levels: int = 1, mid_channels: int = 64, **kwargs):
        super().__init__()
        self.num_levels = num_levels
        self.mid_channels = mid_channels

        self.z = Parameter(torch.zeros(1, mid_channels, dtype=torch.float32), requires_grad=True)
        init.xavier_normal_(self.z)

        self.fc1 = nn.ModuleList([Linear(mid_channels, mid_channels, **kwargs) for _ in range(num_levels)])
        self.fc2 = nn.ModuleList([Linear(mid_channels, 3*10,         **kwargs) for _ in range(num_levels)])
        self.act = F.elu

        if "init_weights" not in kwargs:
            for fc in self.fc2:  # type: ignore
                fc: Linear
                init.xavier_normal_(fc.weight, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x

        for i in range(self.num_levels):
            w: torch.Tensor
            w = self.fc2[i](self.act(self.fc1[i](self.z)))  # (1, 3*10)
            w = w.reshape(1, 3, 10)                         # (1, 3, 10)
            w = w.swapaxes(-2, -1)                          # (1, 10, 3)
            r = apply_quadratic_function(y, w)              # (*, 3, H, W)
            y = y + r

        return y


class GlobalFeatureNet(nn.Module):
    """ Estimate global feature from input image.

    References:
        - Schwartz, E., et al. (2018). "DeepISP: Toward learning an end-to-end image processing pipeline." IEEE TIP 28(2): 912-923.

    Args:
        num_blocks: number of blocks.
        num_channels: number of channels in each block and output.
        num_bins: number of bins in soft histogram.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.

    Outputs:
        output tensor, `(*, num_channels)`.
    """

    def __init__(self, num_blocks: int = 3, num_channels: int = 64, **kwargs):
        super().__init__()
        assert num_blocks >= 1, f"`num_blocks` must be positive, got {num_blocks}"

        self.num_blocks = num_blocks
        self.num_channels = num_channels

        self.convs = nn.ModuleList(
            [Conv2d(3*(1+2+4+8+16), num_channels, 3, stride=2, padding=0, **kwargs)] +
            [Conv2d(num_channels,   num_channels, 3, stride=2, padding=0, **kwargs)
             for _ in range(num_blocks-1)]
        )
        self.act = F.elu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim >= 3, f"Expected color image, got {x.shape}-dim"

        grad    = KF.spatial_gradient(x).flatten(-4, -3)            # (*, 3*2, H, W)
        hist_4  = soft_histogram(x, 4).flatten(-4, -3)              # (*, 3*4, H, W)
        hist_8  = soft_histogram(x, 8).flatten(-4, -3)              # (*, 3*8, H, W)
        hist_16 = soft_histogram(x, 16).flatten(-4, -3)             # (*, 3*16, H, W)

        w = torch.cat([x, grad, hist_4, hist_8, hist_16], dim=-3)   # (*, 3*(1+2+4+8+16), H, W)
        for conv in self.convs:
            w = self.pool(self.act(conv(w)))                        # (*, num_channels, H/2^i, W/2^i)
        w = w.mean(dim=(-2, -1))                                    # (*, num_channels)
        assert w.shape == (*x.shape[:-3], self.num_channels)

        return w


class QuadraticNet(nn.Module):
    """ Global tone mapping network using quadratic functions only.

    References:
        - Schwartz, E., et al. (2018). "DeepISP: Toward learning an end-to-end image processing pipeline." IEEE TIP 28(2): 912-923.

    Args:
        num_levels: number of applying quadratic function.
        num_blocks: number of blocks in global feature network.
        mid_channels: number of channels in the middle layers.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, num_levels: int = 1, num_blocks: int = 3, mid_channels: int = 64, **kwargs):
        super().__init__()

        self.num_levels = num_levels
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels

        self.featnet = GlobalFeatureNet(num_blocks, mid_channels, **kwargs)
        self.fc1 = nn.ModuleList([Linear(mid_channels, mid_channels, **kwargs) for _ in range(num_levels)])
        self.fc2 = nn.ModuleList([Linear(mid_channels, 3*10,         **kwargs) for _ in range(num_levels)])
        self.act = F.elu

        if "init_weights" not in kwargs:
            for fc in self.fc2:  # type: ignore
                fc: Linear
                init.xavier_normal_(fc.weight, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x

        f = self.featnet(x)                             # (*, mid_channels)
        for i in range(self.num_levels):
            w: torch.Tensor
            w = self.fc2[i](self.act(self.fc1[i](f)))   # (*, 3*10)
            w = w.reshape(*x.shape[:-3], 3, 10)         # (*, 3, 10)
            w = w.swapaxes(-2, -1)                      # (*, 10, 3)
            r = apply_quadratic_function(y, w)          # (*, 3, H, W)
            y = y + r

        return y


class QuadraticCRFNet(nn.Module):
    """ Global tone mapping network using quadratic functions and CRF.

    Args:
        num_blocks: number of blocks in global feature network.
        mid_channels: number of channels in the middle layers.
        composition: composition of tone mapping operators.
        inverse: whether to apply inverse EMoR or not.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, num_blocks: int = 3, mid_channels: int = 64,
                 composition: str = "gtgtg", inverse: bool = False, **kwargs):
        super().__init__()
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.composition = composition

        self.emor = data.emor.EMoR(inverse)
        self.featnet = GlobalFeatureNet(num_blocks, mid_channels, **kwargs)
        self.fc = nn.ModuleList([
            Linear(mid_channels, 3*10 if c == "t" else 11 if c == "r" else 1, **kwargs)
            for c in composition
        ])
        self.act = F.elu

        if "init_weights" not in kwargs:
            for fc in self.fc:  # type: ignore
                fc: Linear
                init.xavier_normal_(fc.weight, 0.1)

    def get_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        f = self.featnet(x)  # (*, mid_channels)
        return [self.fc[i](f) for i, c in enumerate(self.composition)]

    def _forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        y = x

        for i, c in enumerate(self.composition):
            w = weights[i]
            match c:
                case "g":
                    w = w.exp()                            # (*, 1)
                    y = apply_gamma_correction(y, w)
                case "t":
                    w = w.reshape(*x.shape[:-3], 3, 10)    # (*, 3, 10)
                    w = w.swapaxes(-2, -1)                 # (*, 10, 3)
                    y = y + apply_quadratic_function(y, w)
                case "r":
                    w = self.emor(w)                       # (*, 1024)
                    y = apply_response_function(y, w)
                case _:
                    raise ValueError(f"Unknown composition: {c}")

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.get_weights(x)
        return self._forward(x, weights)


class DirectQuadraticCRFNet(nn.Module):
    """ Directly learn quadratic functions and CRF.

    Args:
        composition: composition of tone mapping operators.
        inverse: whether to apply inverse EMoR or not.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, composition: str = "gtgtg", inverse: bool = False, **kwargs):
        super().__init__()
        self.composition = composition

        self.emor = data.emor.EMoR(inverse)
        self.params = nn.ParameterList([
            Parameter(torch.zeros((1, 3*10 if c == "t" else 11 if c == "r" else 1), **kwargs))
            for c in composition
        ])

        init_weights = kwargs.get("init_weights", init.xavier_normal_)
        for p in self.params:
            init_weights(p, 0.1)

    def get_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        weights: list[torch.Tensor] = [p for p in self.params]
        weights = [w.repeat(*x.shape[:-3], 1) for w in weights]
        return weights

    def _forward(self, x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
        y = x

        for i, c in enumerate(self.composition):
            w = weights[i]
            match c:
                case "g":
                    w = w.exp()                            # (*, 1)
                    y = apply_gamma_correction(y, w)
                case "t":
                    w = w.reshape(*x.shape[:-3], 3, 10)    # (*, 3, 10)
                    w = w.swapaxes(-2, -1)                 # (*, 10, 3)
                    y = y + apply_quadratic_function(y, w)
                case "r":
                    w = self.emor(w)                       # (*, 1024)
                    y = apply_response_function(y, w)
                case _:
                    raise ValueError(f"Unknown composition: {c}")

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.get_weights(x)
        return self._forward(x, weights)


class HyperQuadraticCRFNet(nn.Module):
    """ Hypernetwork for `QuadraticCRFNet`.

    Args:
        target: target network.
        num_params: dimension of parameters.
        num_blocks: number of blocks in the hypernetwork.
        mid_channels: number of channels in the middle layers of the hypernetwork.
        inverse: whether to apply inverse EMoR or not.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        h: hyperparameters, `(*, num_params)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, target: QuadraticCRFNet, num_params: int = 256,
                 num_blocks: int = 3, mid_channels: int = 64, use_attn: bool = True, **kwargs):
        super().__init__()
        self.target = target
        self.num_params = num_params
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.use_attn = use_attn

        fc1: list[nn.Module] = []
        fc1.append(Linear(num_params, mid_channels, **kwargs))
        fc1.append(nn.ELU())
        for _ in range(1, num_blocks):
            fc1.append(Linear(mid_channels, mid_channels, **kwargs))
            fc1.append(nn.ELU())
        self.fc1 = nn.Sequential(*fc1)

        self.fc2 = nn.ModuleList([
            Linear(mid_channels, 3*10 if c == "t" else 11 if c == "r" else 1, **kwargs)
            for c in target.composition
        ])
        if "init_weights" not in kwargs:
            for fc in self.fc2:  # type: ignore
                fc: Linear
                init.xavier_normal_(fc.weight, 0.1)

    def get_weights(self, x: torch.Tensor, h: torch.Tensor) -> list[torch.Tensor]:
        weights = self.target.get_weights(x)

        feat = self.fc1(h)  # (*, mid_channels)
        for i, c in enumerate(self.target.composition):
            mod = self.fc2[i](feat)
            if self.use_attn:
                weights[i] = weights[i] + weights[i] * mod
            else:
                weights[i] = weights[i] + mod

        return weights

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        weights = self.get_weights(x, h)
        return self.target._forward(x, weights)


class HyperDirectQuadraticCRFNet(nn.Module):
    """ Hypernetwork for `DirectQuadraticCRFNet`.

    Args:
        target: target network.
        num_params: dimension of parameters.
        num_blocks: number of blocks in the hypernetwork.
        mid_channels: number of channels in the middle layers of the hypernetwork.
        inverse: whether to apply inverse EMoR or not.

    Inputs:
        x: input tensor, `(*, 3, H, W)`.
        h: hyperparameters, `(*, num_params)`.

    Outputs:
        output tensor, `(*, 3, H, W)`.
    """

    def __init__(self, target: DirectQuadraticCRFNet, num_params: int = 256,
                 num_blocks: int = 3, mid_channels: int = 64, use_attn: bool = True, **kwargs):
        super().__init__()
        self.target = target
        self.num_params = num_params
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.use_attn = use_attn

        fc1: list[nn.Module] = []
        fc1.append(Linear(num_params, mid_channels, **kwargs))
        fc1.append(nn.ELU())
        for _ in range(1, num_blocks):
            fc1.append(Linear(mid_channels, mid_channels, **kwargs))
            fc1.append(nn.ELU())
        self.fc1 = nn.Sequential(*fc1)

        self.fc2 = nn.ModuleList([
            Linear(mid_channels, 3*10 if c == "t" else 11 if c == "r" else 1, **kwargs)
            for c in target.composition
        ])
        if "init_weights" not in kwargs:
            for fc in self.fc2:  # type: ignore
                fc: Linear
                init.xavier_normal_(fc.weight, 0.1)

    def get_weights(self, x: torch.Tensor, h: torch.Tensor) -> list[torch.Tensor]:
        weights = self.target.get_weights(x)

        feat = self.fc1(h)  # (*, mid_channels)
        for i, c in enumerate(self.target.composition):
            mod = self.fc2[i](feat)
            if self.use_attn:
                weights[i] = weights[i] + weights[i] * mod
            else:
                weights[i] = weights[i] + mod

        return weights

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        weights = self.get_weights(x, h)
        return self.target._forward(x, weights)
