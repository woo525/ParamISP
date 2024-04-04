from typing import Callable
import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .base import Conv2d, Linear


def position_map(x: torch.Tensor, full_size: list[tuple[int, int]], top_left: list[tuple[int, int]]) -> torch.Tensor:
    """ Compute position map of each pixels.

    Args:
        x: input tensor, `(N, C, H, W)`.
        full_size: full size of each image, `[(H, W), ...]`.
        top_left: top-left position of each image, `[(y, x), ...]`.

    Returns:
        output tensor, `(N, 4, H, W)`.
    """
    assert x.ndim == 4, f"Expected batch image, got {x.shape}-dim"
    assert len(full_size) == len(top_left), \
        f"Expected same length of `full_size` and `top_left`, got {len(full_size)} and {len(top_left)}"

    N = len(full_size)
    H, W = x.shape[-2:]
    y_i, x_i = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    y_i = y_i.to(device=x.device, dtype=x.dtype)
    x_i = x_i.to(device=x.device, dtype=x.dtype)

    y_abs = torch.stack([y_i + top_left[b][0] - (full_size[b][0]//2) for b in range(N)])
    x_abs = torch.stack([x_i + top_left[b][1] - (full_size[b][1]//2) for b in range(N)])
    y_rel = torch.stack([y_abs[b] / full_size[b][0] for b in range(N)])
    x_rel = torch.stack([x_abs[b] / full_size[b][1] for b in range(N)])

    return torch.stack([y_abs, x_abs, y_rel, x_rel], dim=-3)


class GaussianMask(nn.Module):
    """ Learnable Gaussian mask layer.

    Inputs:
        x: input tensor, `(N, C, H, W)`.
        full_size: full size of each image, `[(H, W), ...]`.
        top_left: top-left position of each image, `[(y, x), ...]`.

    Outputs:
        - output: output tensor, `(N, C, H, W)`.
    """

    def __init__(self):
        super().__init__()
        self._sigma = nn.parameter.Parameter(torch.tensor(+.02), requires_grad=True)
        self._scale = nn.parameter.Parameter(torch.tensor(-.02), requires_grad=True)
        self._alpha = nn.parameter.Parameter(torch.tensor(-.09), requires_grad=True)

    def _forward(
        self, x: torch.Tensor, full_size: list[tuple[int, int]], top_left: list[tuple[int, int]],
        coef_sigma: torch.Tensor, coef_scale: torch.Tensor, coef_alpha: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        assert x.ndim == 4, f"Expected batch image, got {x.shape}-dim"
        assert len(full_size) == len(top_left), \
            f"Expected same length of `full_size` and `top_left`, got {len(full_size)} and {len(top_left)}"

        pos = position_map(x, full_size, top_left)             # (N, 4, H, W)
        r_abs = torch.norm(pos[:, 0:2], dim=-3, keepdim=True)  # (N, 1, H, W)
        r_rel = torch.norm(pos[:, 2:4], dim=-3, keepdim=True)  # (N, 1, H, W)

        sigma = 100. * coef_sigma
        scale = torch.sigmoid(100. * coef_scale)
        alpha = torch.sigmoid(100. * coef_alpha)

        r = alpha * r_abs + (1 - alpha) * r_rel                        # (N, 1, H, W)
        mask = torch.exp(-r**2 / (2*sigma**2)) * scale + (1. - scale)  # (N, 1, H, W)

        if inverse:
            y = x / mask
        else:
            y = x * mask

        return y

    def forward(
        self, x: torch.Tensor, full_size: list[tuple[int, int]], top_left: list[tuple[int, int]],
        inverse: bool = False,
    ) -> torch.Tensor:
        return self._forward(
            x, full_size, top_left,
            self._sigma, self._scale, self._alpha,
            inverse,
        )


class HyperGaussianMask(nn.Module):
    """ Hypernetwork for `GaussianMask`.

    Args:
        target: target network.
        num_params: dimension of parameters.
        num_blocks: number of blocks in the hypernetwork.
        mid_channels: number of channels in the middle layers of the hypernetwork.

    Inputs:
        x: input tensor, `(N, C, H, W)`.
        full_size: full size of each image, `[(H, W), ...]`.
        top_left: top-left position of each image, `[(y, x), ...]`.

    Outputs:
        - output: output tensor, `(N, C, H, W)`.
    """

    def __init__(self, target: GaussianMask, num_params: int = 256,
                 num_blocks: int = 3, mid_channels: int = 64, **kwargs):
        super().__init__()
        self.target = target
        self.num_params = num_params
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels

        fc: list[nn.Module] = [Linear(num_params, mid_channels, **kwargs)]
        for _ in range(1, num_blocks):
            fc.append(nn.ELU())
            fc.append(Linear(mid_channels, mid_channels, **kwargs))
        fc.append(nn.ELU())
        last_fc = Linear(mid_channels, 3, **kwargs)
        if "init_weights" not in kwargs:
            init.xavier_normal_(last_fc.weight, 0.02)
        fc.append(last_fc)
        self.fc = nn.Sequential(*fc)

    def coefficients(self, h: torch.Tensor):
        w = self.fc(h)
        sigma = self.target._sigma
        scale = self.target._scale
        alpha = self.target._alpha
        sigma = sigma + sigma * w[:, 0:1]
        scale = scale + scale * w[:, 1:2]
        alpha = alpha + alpha * w[:, 2:3]
        return sigma, scale, alpha

    def forward(
        self, x: torch.Tensor, h: torch.Tensor,
        full_size: list[tuple[int, int]],
        top_left: list[tuple[int, int]],
        inverse: bool = False,
    ) -> torch.Tensor:
        coef_sigma, coef_scale, coef_alpha = self.coefficients(h)
        return self.target._forward(
            x, full_size, top_left,
            coef_sigma, coef_scale, coef_alpha,
            inverse,
        )
